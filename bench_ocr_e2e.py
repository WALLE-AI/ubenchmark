#!/usr/bin/env python3
"""文档解析 VLM (MinerU2.5-Pro / TeleOCR) 的端到端页级压测。

与 bench_ocr_vlm.py (合成图片) 和 bench_ocr_pdf.py (真实 PDF，但单请求整页通用 prompt)
的区别: 这里复刻两个模型的**官方两阶段推理协议**，以"页/秒 (FPS)"为主指标。

  阶段1 layout: 整页 resize 到 1036x1036，发模型原生 layout prompt，拿回版面框
  阶段2 block : 按版面框裁原图，每块发一次对应类型的识别 prompt

一次"页面任务" = 1 次 layout 请求 + N 次 block 请求(并发)，N 由该页版面复杂度决定。
这才是生产里一页文档的真实开销；单发一次整页通用 prompt 会让输出只有几十 token、
延迟只有几百毫秒，测出的并发数严重虚高。

压测数据集从真实语料 (/home/dataset-s3-0/datasets/PDFs/data) 按 文件大小 x 页数
分层抽样得到，固化成 manifest JSON 以便复现。

依赖: PyMuPDF / Pillow / aiohttp / opencv-python，用 docparserEnv 运行:
  PY=/home/dataset-s3-0/gaojing/anaconda3/envs/docparserEnv/bin/python

用法:
  # 1. 构建压测数据集 (5 档 x 3 篇 x 3 页)
  $PY bench_ocr_e2e.py build-dataset --output ocr_bench_dataset.json

  # 2. 跑压测。重定向到文件时记得加 -u，否则 stdout 被块缓冲，看不到进度。
  $PY -u bench_ocr_e2e.py run --dataset ocr_bench_dataset.json \
      --concurrency-levels 1,2,4,8,16 \
      --output ocr_e2e_report.json --html ocr_e2e_report.html | tee logs/bench_e2e.log
"""

import argparse
import asyncio
import base64
import io
import itertools
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter

import aiohttp
import cv2
import fitz
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ocr_pdf import (  # noqa: E402
    PAGE_TIERS,
    PROFILES,
    SIZE_TIERS,
    page_tier_of,
    scan_catalog,
    size_tier_of,
)
from bench_ocr_vlm import percentile  # noqa: E402
from ocr_report_html import write_report as write_html  # noqa: E402

DEFAULT_PDF_DIR = "/home/dataset-s3-0/datasets/PDFs/data"
DEFAULT_DATASET = "ocr_bench_dataset.json"

# 两个模型的官方客户端都用这套图片预处理常量:
#   mineru_vl_utils/mineru_client.py:278  /  TeleOCR/vlm_utils/TeleOCR_client.py:392
LAYOUT_IMAGE_SIZE = (1036, 1036)
MIN_IMAGE_EDGE = 28
MAX_IMAGE_EDGE_RATIO = 50

# 两个模型的 prepare_for_extract 都跳过这几类块 (图片块不做 OCR，list 是容器块)
SKIP_BLOCK_TYPES = {"image", "list", "equation_block"}


# --------------------------------------------------------------------------
# 模型适配层: 封装各自的原生两阶段协议
# --------------------------------------------------------------------------

MINERU_LAYOUT_RE = re.compile(
    r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|>"
    r"<\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"
)
TELEOCR_LAYOUT_RE = re.compile(r"^<box:([\d\s]+)><label:(\w+)><([^>]+)>$")

MINERU_ANGLE_MAPPING = {
    "<|rotate_up|>": 0, "<|rotate_right|>": 90,
    "<|rotate_down|>": 180, "<|rotate_left|>": 270,
}
TELEOCR_ANGLE_MAPPING = {"up": 0, "right": 90, "down": 180, "left": 270}


class Block:
    __slots__ = ("type", "points", "angle")

    def __init__(self, btype, points, angle):
        self.type = btype
        self.points = points  # [(x, y), ...] 归一化 0-1，2 点=矩形，>2 点=多边形
        self.angle = angle


class Adapter:
    """一个模型的原生推理协议。"""

    name = ""
    layout_prompt = ""
    block_prompts = {}
    default_block_prompt = ""
    # block 类型 -> (presence_penalty, frequency_penalty)，其余采样参数两模型一致
    block_penalties = {}
    default_penalties = (1.0, 0.05)

    def parse_layout(self, text):
        raise NotImplementedError

    def prompt_for(self, btype):
        return self.block_prompts.get(btype) or self.default_block_prompt

    def penalties_for(self, btype):
        return self.block_penalties.get(btype, self.default_penalties)


class MinerUAdapter(Adapter):
    """MinerU2.5-Pro，协议见 mineru_vl_utils/mineru_client.py:41 DEFAULT_PROMPTS。"""

    name = "mineru"
    layout_prompt = "\nLayout Detection:"
    block_prompts = {
        "table": "\nTable Recognition:",
        "equation": "\nFormula Recognition:",
    }
    default_block_prompt = "\nText Recognition:"
    block_penalties = {"table": (1.0, 0.005)}

    def parse_layout(self, text):
        blocks = []
        for line in text.split("\n"):
            m = MINERU_LAYOUT_RE.match(line.strip())
            if not m:
                continue
            coords = [int(v) for v in m.groups()[:4]]
            if any(c < 0 or c > 1000 for c in coords):
                continue
            x1, y1, x2, y2 = coords
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x1 == x2 or y1 == y2:
                continue
            angle = next(
                (a for tok, a in MINERU_ANGLE_MAPPING.items() if tok in m.group(6)), 0
            )
            blocks.append(
                Block(m.group(5).lower(), [(x1 / 1000, y1 / 1000), (x2 / 1000, y2 / 1000)], angle)
            )
        return blocks


class TeleOCRAdapter(Adapter):
    """TeleOCR，协议见 TeleOCR/vlm_utils/TeleOCR_client.py:39 LAYOUT_PROMPTS / :44 DEFAULT_PROMPTS。

    config.py LAYOUT_MODE 默认 "Detection"，对应 layout prompt "Analyze the image layout."
    """

    name = "teleocr"
    layout_prompt = "\nAnalyze the image layout."
    block_prompts = {
        "text": "\nPlease output the text content from the image.",
        "table": "\nThis is the image of a table. Please output the table in OTSL format.",
        "equation": "\nPlease write out the expression of the formula in the image using LaTeX format.",
        "code": "\nThe image contains a code snippet, please output the parsing result.",
        "seal": "\nSeal Recognition:",
        "char": "\nThis is a scientific figure. Please extract the table implied by this figure.",
    }
    default_block_prompt = "\nPlease output the text content from the image."
    block_penalties = {
        "table": (1.0, 0.005), "char": (1.0, 0.005), "seal": (1.0, 0.005),
    }

    def parse_layout(self, text):
        blocks = []
        for line in text.split("\n"):
            m = TELEOCR_LAYOUT_RE.match(line.strip())
            if not m:
                continue
            nums = [int(v) for v in m.group(1).split()]
            if len(nums) < 4 or len(nums) % 2:
                continue
            if any(c < 0 or c > 1000 for c in nums):
                continue
            pts = [(nums[i] / 1000, nums[i + 1] / 1000) for i in range(0, len(nums), 2)]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if min(xs) == max(xs) or min(ys) == max(ys):
                continue
            angle = next(
                (a for tok, a in TELEOCR_ANGLE_MAPPING.items() if tok in m.group(3)), 0
            )
            blocks.append(Block(m.group(2).lower(), pts, angle))
        return blocks


ADAPTERS = {"mineru": MinerUAdapter(), "teleocr": TeleOCRAdapter()}

DEFAULT_TARGETS = [
    {"name": "mineru", "base_url": "http://127.0.0.1:8003",
     "model": "mineru2.5-pro-1.2b", "adapter": "mineru", "gpu": 0},
    {"name": "teleocr", "base_url": "http://127.0.0.1:8004",
     "model": "teleocr", "adapter": "teleocr", "gpu": 1},
]


# --------------------------------------------------------------------------
# 图片预处理 (复刻两个官方客户端)
# --------------------------------------------------------------------------

def resize_by_need(image):
    """mineru_client.py:107 resize_by_need —— 极端长宽比补白 + 最短边不足则放大。"""
    edge_ratio = max(image.size) / max(min(image.size), 1)
    if edge_ratio > MAX_IMAGE_EDGE_RATIO:
        width, height = image.size
        if width > height:
            new_w, new_h = width, int(np.ceil(width / MAX_IMAGE_EDGE_RATIO))
        else:
            new_w, new_h = int(np.ceil(height / MAX_IMAGE_EDGE_RATIO)), height
        canvas = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
        canvas.paste(image, (int((new_w - width) / 2), int((new_h - height) / 2)))
        image = canvas
    if min(image.size) < MIN_IMAGE_EDGE:
        scale = MIN_IMAGE_EDGE / min(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.BICUBIC,
        )
    return image


def prepare_layout_image(page_image):
    return page_image.resize(LAYOUT_IMAGE_SIZE, Image.Resampling.BICUBIC)


def crop_block(page_image, page_np, block):
    """按 block 几何裁剪原页。矩形直接 crop，多边形走 boundingRect + fillPoly 掩膜
    (复刻 TeleOCR_client.py:245 起的逻辑)。"""
    width, height = page_image.size
    pts = np.array(
        [(p[0] * width, p[1] * height) for p in block.points], dtype=np.float32
    ).astype(np.int32)

    if len(pts) == 2:
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        crop = page_image.crop((int(x1), int(y1), int(x2), int(y2)))
    else:
        x, y, w, h = cv2.boundingRect(pts)
        if w < 1 or h < 1:
            return None
        crop_np = page_np[y:y + h, x:x + w]
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = pts.copy()
        shifted[:, 0] -= x
        shifted[:, 1] -= y
        cv2.fillPoly(mask, [shifted], 255)
        crop = Image.fromarray(cv2.bitwise_and(crop_np, crop_np, mask=mask))

    if crop.width < 1 or crop.height < 1:
        return None
    if block.angle in (90, 180, 270):
        crop = crop.rotate(block.angle, expand=True)
    return resize_by_need(crop)


def to_data_uri(image):
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# 数据集构建: 按 文件大小 x 页数 分层抽样
# --------------------------------------------------------------------------

def build_dataset(pdf_dir, docs_per_profile, pages_per_doc, dpi, seed, max_attempts):
    print(f"扫描语料目录: {pdf_dir}")
    catalog = scan_catalog(pdf_dir)
    print(f"共 {len(catalog)} 个 PDF，按文件大小分档:")
    tier_counts = {}
    for tier_name, lo, hi in SIZE_TIERS:
        cnt = sum(1 for _, s in catalog if lo <= s < hi)
        tier_counts[tier_name] = cnt
        hi_str = "∞" if hi == float("inf") else f"{hi / 1024 / 1024:.0f}MB"
        print(f"  {tier_name:<8s} [{lo / 1024 / 1024:.2f}MB, {hi_str})  count={cnt}")

    rng = random.Random(seed)
    by_size_tier = {}
    for path, size in catalog:
        by_size_tier.setdefault(size_tier_of(size), []).append((path, size))

    print(f"\n分层抽样: 每档 {docs_per_profile} 篇 x 每篇 {pages_per_doc} 页 (dpi={dpi})")
    entries = []
    for profile_name, size_tier, page_tier in PROFILES:
        candidates = list(by_size_tier.get(size_tier, []))
        rng.shuffle(candidates)
        picked = 0
        for path, size in candidates[:max_attempts]:
            if picked >= docs_per_profile:
                break
            try:
                doc = fitz.open(path)
            except Exception:  # noqa: BLE001
                continue
            try:
                if doc.needs_pass:
                    continue
                n_pages = len(doc)
                if page_tier_of(n_pages) != page_tier:
                    continue
                page_indices = pick_page_indices(n_pages, pages_per_doc)
                # 验证这些页真能渲染出来，避免压测中途炸掉
                try:
                    for idx in page_indices:
                        doc.load_page(idx).get_pixmap(dpi=dpi)
                except Exception:  # noqa: BLE001
                    continue
                try:
                    toc = doc.get_toc()
                    chapters = sum(1 for e in toc if e[0] == 1) or None
                except Exception:  # noqa: BLE001
                    chapters = None
                entries.append({
                    "profile": profile_name, "size_tier": size_tier, "page_tier": page_tier,
                    "path": path, "file": os.path.basename(path), "size_bytes": size,
                    "pages": n_pages, "chapters": chapters,
                    "page_indices": page_indices, "dpi": dpi,
                })
                picked += 1
            finally:
                doc.close()
        status = "OK" if picked == docs_per_profile else f"只找到 {picked} 篇"
        print(f"  {profile_name:<18s} (size={size_tier}, pages={page_tier}): {status}")
        for e in entries[-picked:] if picked else []:
            print(f"      {e['file'][:56]:<56s} {e['size_bytes'] / 1024 / 1024:>7.2f}MB "
                  f"{e['pages']:>4d}页 章节={e['chapters']} 取页={e['page_indices']}")

    return {
        "pdf_dir": pdf_dir, "seed": seed, "dpi": dpi,
        "docs_per_profile": docs_per_profile, "pages_per_doc": pages_per_doc,
        "catalog_total": len(catalog), "catalog_size_tiers": tier_counts,
        "entries": entries,
    }


def pick_page_indices(n_pages, want):
    """从文档内均匀取页: 首 / 中 / 尾 ... 页数不足时去重。"""
    if n_pages <= want:
        return list(range(n_pages))
    if want == 1:
        return [0]
    step = (n_pages - 1) / (want - 1)
    idx = sorted({int(round(i * step)) for i in range(want)})
    # 去重后可能不足，从前往后补
    for i in range(n_pages):
        if len(idx) >= want:
            break
        if i not in idx:
            idx.append(i)
    return sorted(idx)[:want]


# --------------------------------------------------------------------------
# HTTP 请求
# --------------------------------------------------------------------------

def build_request(model, prompt, data_uri, max_tokens, penalties):
    presence, frequency = penalties
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.01,
        "top_k": 1,
        "repetition_penalty": 1.0,
        "presence_penalty": presence,
        "frequency_penalty": frequency,
        # MinerU 的 layout 输出用 <|box_start|> 等特殊 token 分隔字段，
        # vLLM 默认 skip_special_tokens=True 会剥掉它们导致无法解析。
        # 官方客户端 no_repeat_ngram_size=100，vLLM OpenAI 接口不支持，此处省略。
        "skip_special_tokens": False,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }],
    }


async def post_chat(session, url, payload, timeout_s, retries=1):
    """返回 (ok, latency_s, text, prompt_tokens, completion_tokens, error)。

    仅对连接层错误重试 (keep-alive 竞争: 服务端超时关连接的瞬间客户端正好复用它，
    表现为 Broken pipe / Connection reset)。生产客户端都这么做，不重试会把这类
    传输层抖动误记成模型的错误率。模型返回的 4xx/5xx 不重试，那是真实失败。
    """
    t0 = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
            ) as resp:
                body = await resp.json()
                dt = time.perf_counter() - t0
                if resp.status != 200:
                    return False, dt, "", 0, 0, f"HTTP {resp.status}: {str(body)[:160]}"
                usage = body.get("usage") or {}
                return (True, dt, body["choices"][0]["message"]["content"],
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0), None)
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError) as e:
            if attempt >= retries:
                return (False, time.perf_counter() - t0, "", 0, 0,
                        f"{type(e).__name__}: {e} (重试 {retries} 次后仍失败)")
        except Exception as e:  # noqa: BLE001
            return False, time.perf_counter() - t0, "", 0, 0, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# 预飞: 对每页跑一次 layout，裁好所有 block 图并缓存成 data URI
# --------------------------------------------------------------------------

class PreparedPage:
    """一页的完整重放素材: layout 整页图 + 已裁好的 N 个 block 图。

    存 PIL 图而非 data URI，是为了在每个并发档开跑前重新打水印+编码 (见 materialize_tasks)。
    """

    __slots__ = ("entry", "page_index", "layout_image", "blocks", "page_px")

    def __init__(self, entry, page_index, layout_image, blocks, page_px):
        self.entry = entry
        self.page_index = page_index
        self.layout_image = layout_image
        self.blocks = blocks        # [(btype, prompt, PIL.Image, penalties)]
        self.page_px = page_px


class PageTask:
    """一个页面任务的成品请求素材 (data URI 已编码好)。"""

    __slots__ = ("layout_uri", "blocks", "profile")

    def __init__(self, layout_uri, blocks, profile):
        self.layout_uri = layout_uri
        self.blocks = blocks        # [(prompt, data_uri, penalties)]
        self.profile = profile


def stamp(image, nonce):
    """在左上角打一个由 nonce 决定的像素水印。

    vLLM 的前缀缓存对多模态输入按内容哈希，重放同一张图会整段命中缓存 —— 实测
    重放第二遍时命中率高达 96%，吞吐被严重高估。生产里每页都是新的，所以这里让
    每个请求实例的图片字节都唯一，保证必然 cache miss，读数才对得上真实产能。
    单像素改动对识别结果的影响可忽略。
    """
    out = image.copy()
    out.putpixel((0, 0), (nonce & 0xFF, (nonce >> 8) & 0xFF, (nonce >> 16) & 0xFF))
    return out


def materialize_tasks(pages, n_tasks, counter, cache_busting):
    """为一个并发档生成 n_tasks 个页面任务的请求素材。

    编码开销全部落在这里，在计时区间之外 —— 否则 16 并发 x 12 块的 PNG/base64
    编码会把客户端自己压成瓶颈，测的就不是服务端了。
    """
    tasks = []
    for i in range(n_tasks):
        page = pages[i % len(pages)]
        if cache_busting:
            nonce = next(counter)
            layout_uri = to_data_uri(stamp(page.layout_image, nonce))
            blocks = [
                (prompt, to_data_uri(stamp(img, nonce)), penalties)
                for _btype, prompt, img, penalties in page.blocks
            ]
        else:
            layout_uri = to_data_uri(page.layout_image)
            blocks = [
                (prompt, to_data_uri(img), penalties)
                for _btype, prompt, img, penalties in page.blocks
            ]
        tasks.append(PageTask(layout_uri, blocks, page.entry["profile"]))
    return tasks


def render_page(path, page_index, dpi):
    doc = fitz.open(path)
    try:
        pix = doc.load_page(page_index).get_pixmap(dpi=dpi)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()


async def preflight(session, target, adapter, dataset, max_tokens, timeout_s, block_cap):
    """串行跑一遍 layout，拿到每页真实的版面块并裁好图。不计入压测指标。"""
    print(f"\n[{target['name']}] 预飞: 对 {len(dataset['entries'])} 篇文档的页面跑 layout 并裁块...")
    url = f"{target['base_url']}/v1/chat/completions"
    prepared = []
    failures = []

    for entry in dataset["entries"]:
        for page_index in entry["page_indices"]:
            page_image = render_page(entry["path"], page_index, entry["dpi"])
            page_np = np.array(page_image)
            layout_image = prepare_layout_image(page_image)

            payload = build_request(
                target["model"], adapter.layout_prompt, to_data_uri(layout_image),
                max_tokens, (0.0, 0.0),
            )
            ok, _, text, _, _, err = await post_chat(session, url, payload, timeout_s)
            if not ok:
                failures.append(f"{entry['file']} p{page_index}: layout 请求失败 {err}")
                continue

            blocks = adapter.parse_layout(text)
            if not blocks:
                failures.append(f"{entry['file']} p{page_index}: layout 未解析出 block")
                continue

            block_items = []
            for block in blocks:
                if block.type in SKIP_BLOCK_TYPES:
                    continue
                crop = crop_block(page_image, page_np, block)
                if crop is None:
                    continue
                block_items.append((
                    block.type,
                    adapter.prompt_for(block.type),
                    crop,
                    adapter.penalties_for(block.type),
                ))
                if block_cap and len(block_items) >= block_cap:
                    break

            if not block_items:
                failures.append(f"{entry['file']} p{page_index}: 无可识别 block (全被跳过)")
                continue

            prepared.append(PreparedPage(
                entry, page_index, layout_image, block_items,
                (page_image.width, page_image.height),
            ))

    by_profile = {}
    for p in prepared:
        by_profile.setdefault(p.entry["profile"], []).append(p)

    total_blocks = sum(len(p.blocks) for p in prepared)
    print(f"[{target['name']}] 预飞完成: {len(prepared)} 页可用, "
          f"共 {total_blocks} 个 block, 平均 {total_blocks / max(len(prepared), 1):.1f} 块/页")
    for profile_name, pages in by_profile.items():
        blocks = sum(len(p.blocks) for p in pages)
        types = Counter(b[0] for p in pages for b in p.blocks)
        print(f"    {profile_name:<18s} {len(pages):>3d}页 {blocks:>4d}块 "
              f"({blocks / len(pages):.1f}块/页)  {dict(types.most_common(5))}")
    for f in failures[:10]:
        print(f"    ! {f}")
    if len(failures) > 10:
        print(f"    ! ... 另有 {len(failures) - 10} 条")

    return by_profile, failures


# --------------------------------------------------------------------------
# 压测执行
# --------------------------------------------------------------------------

class PageOutcome:
    __slots__ = ("ok", "e2e_s", "layout_s", "block_s_max", "n_blocks",
                 "prompt_tokens", "completion_tokens", "errors", "messages")

    def __init__(self):
        self.ok = False
        self.e2e_s = 0.0
        self.layout_s = 0.0
        self.block_s_max = 0.0
        self.n_blocks = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.errors = []
        self.messages = []   # 失败原因原文，用于定性高并发下的错误类型


async def run_one_page(session, url, model, layout_prompt, task, max_tokens,
                       timeout_s, block_sem):
    """一个页面任务: layout 请求 -> (等它回来) -> N 个 block 请求并发。"""
    out = PageOutcome()
    t0 = time.perf_counter()

    payload = build_request(model, layout_prompt, task.layout_uri, max_tokens, (0.0, 0.0))
    ok, dt, _, ptok, ctok, err = await post_chat(session, url, payload, timeout_s)
    out.layout_s = dt
    out.prompt_tokens += ptok
    out.completion_tokens += ctok
    if not ok:
        out.errors.append("layout")
        out.messages.append(f"layout: {err}")
        out.e2e_s = time.perf_counter() - t0
        return out

    async def one_block(item):
        prompt, uri, penalties = item
        payload = build_request(model, prompt, uri, max_tokens, penalties)
        if block_sem is None:
            return await post_chat(session, url, payload, timeout_s)
        async with block_sem:
            return await post_chat(session, url, payload, timeout_s)

    results = await asyncio.gather(*(one_block(b) for b in task.blocks))
    for ok_b, dt_b, _text, ptok_b, ctok_b, err_b in results:
        out.prompt_tokens += ptok_b
        out.completion_tokens += ctok_b
        out.block_s_max = max(out.block_s_max, dt_b)
        if not ok_b:
            out.errors.append("block")
            out.messages.append(f"block: {err_b}")

    out.n_blocks = len(task.blocks)
    out.e2e_s = time.perf_counter() - t0
    out.ok = not out.errors
    return out


async def run_level(target, layout_prompt, tasks, concurrency, max_tokens,
                    timeout_s, block_concurrency, gpu_index, connection_limit):
    """一个并发档: 始终保持 `concurrency` 个页面任务在飞，跑完 tasks。"""
    url = f"{target['base_url']}/v1/chat/completions"
    block_sem = asyncio.Semaphore(block_concurrency) if block_concurrency else None
    cursor = 0
    lock = asyncio.Lock()
    outcomes = []

    sampler = GpuSampler(gpu_index)
    prefix_before = await fetch_prefix_cache_stats(target["base_url"])

    async def worker(session):
        nonlocal cursor
        while True:
            async with lock:
                if cursor >= len(tasks):
                    return
                task = tasks[cursor]
                cursor += 1
            outcomes.append(
                await run_one_page(session, url, target["model"], layout_prompt,
                                   task, max_tokens, timeout_s, block_sem)
            )

    sampler.start()
    # 连接池上限: C=64 x ~15 块/页 会瞬间要开近千条 TCP 连接，单个 uvicorn 扛不住，
    # 表现为 Connection reset by peer —— 那测的是连接层而不是推理算力。
    # 用有界连接池(带 keep-alive 复用)，与生产客户端的行为一致。
    connector = aiohttp.TCPConnector(limit=connection_limit)
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(session) for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    sampler.stop()

    prefix_after = await fetch_prefix_cache_stats(target["base_url"])
    return summarize_level(concurrency, outcomes, wall, sampler.summary(),
                           prefix_delta(prefix_before, prefix_after))


def summarize_level(concurrency, outcomes, wall, gpu, prefix):
    ok = [o for o in outcomes if o.ok]
    e2e = [o.e2e_s for o in ok]
    layout = [o.layout_s for o in outcomes if o.layout_s > 0]
    total_req = sum(1 + o.n_blocks for o in outcomes)
    ptok = sum(o.prompt_tokens for o in outcomes)
    ctok = sum(o.completion_tokens for o in outcomes)
    n_layout_err = sum(1 for o in outcomes if "layout" in o.errors)
    n_block_err = sum(o.errors.count("block") for o in outcomes)
    error_samples = Counter(m for o in outcomes for m in o.messages)

    return {
        "concurrency": concurrency,
        "pages_total": len(outcomes),
        "pages_ok": len(ok),
        "wall_s": wall,
        "fps": len(ok) / wall if wall else 0.0,
        "requests_total": total_req,
        "req_per_s": total_req / wall if wall else 0.0,
        "avg_blocks_per_page": (sum(o.n_blocks for o in outcomes) / len(outcomes)) if outcomes else 0,
        "e2e_p50_ms": percentile(e2e, 50) * 1000 if e2e else None,
        "e2e_p90_ms": percentile(e2e, 90) * 1000 if e2e else None,
        "e2e_p99_ms": percentile(e2e, 99) * 1000 if e2e else None,
        "e2e_max_ms": max(e2e) * 1000 if e2e else None,
        "layout_p50_ms": percentile(layout, 50) * 1000 if layout else None,
        "layout_p99_ms": percentile(layout, 99) * 1000 if layout else None,
        "prompt_tokens_per_s": ptok / wall if wall else 0.0,
        "completion_tokens_per_s": ctok / wall if wall else 0.0,
        "error_rate": 1 - (len(ok) / len(outcomes)) if outcomes else 1.0,
        "layout_errors": n_layout_err,
        "block_errors": n_block_err,
        "error_samples": [{"message": m[:300], "count": n}
                          for m, n in error_samples.most_common(5)],
        "gpu": gpu,
        "prefix_cache": prefix,
    }


# --------------------------------------------------------------------------
# GPU / vLLM 指标采样
# --------------------------------------------------------------------------

class GpuSampler:
    """压测期间按 1s 采样目标 GPU 的显存与利用率。"""

    def __init__(self, gpu_index, interval=1.0):
        self.gpu_index = gpu_index
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.mem = []
        self.util = []

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self.gpu_index}",
                     "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                mem_s, util_s = out.stdout.strip().split(",")
                self.mem.append(float(mem_s))
                self.util.append(float(util_s))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def start(self):
        if self.gpu_index is None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def summary(self):
        if not self.mem:
            return None
        return {
            "samples": len(self.mem),
            "mem_used_mb_max": max(self.mem),
            "mem_used_mb_avg": statistics.mean(self.mem),
            "util_pct_avg": statistics.mean(self.util),
            "util_pct_max": max(self.util),
        }


async def fetch_prefix_cache_stats(base_url):
    """从 vLLM /metrics 抓前缀缓存计数。重放同一批页面会命中前缀缓存，
    把命中率报出来，读数才有据可依。"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/metrics", timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                text = await resp.text()
    except Exception:  # noqa: BLE001
        return None
    stats = {}
    for line in text.split("\n"):
        if line.startswith("#") or "prefix_cache" not in line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        key = "hits" if "hits" in parts[0] else "queries" if "queries" in parts[0] else None
        if key:
            try:
                stats[key] = stats.get(key, 0.0) + float(parts[1])
            except ValueError:
                pass
    return stats or None


def prefix_delta(before, after):
    if not before or not after:
        return None
    q = after.get("queries", 0) - before.get("queries", 0)
    h = after.get("hits", 0) - before.get("hits", 0)
    if q <= 0:
        return None
    return {"queries": q, "hits": h, "hit_rate": h / q}


# --------------------------------------------------------------------------
# 建议最大支撑并发
# --------------------------------------------------------------------------

def pick_max_concurrency(levels, p99_threshold_ms, error_rate_threshold, fps_tolerance):
    """健康档 = p99 单页 e2e 在阈值内 且 错误率达标。

    在健康档里找"拐点": 先取峰值吞吐，再回头取**最小**的那个已经跑到峰值
    (1 - fps_tolerance) 的并发 —— 再往上加并发只换来不到 tolerance 的吞吐，
    却要线性地付延迟。

    不用"逐档比较增益、首次跌破即停"的写法: 压测本身有抖动，单个噪声点
    (比如 2.07→2.13 只涨 2.9%) 会把曲线提前截断，哪怕后面吞吐还在继续爬。
    对峰值取容差对单点抖动不敏感。
    """

    def is_healthy(r):
        return (r["error_rate"] <= error_rate_threshold
                and r["e2e_p99_ms"] is not None
                and r["e2e_p99_ms"] <= p99_threshold_ms)

    levels = sorted(levels, key=lambda r: r["concurrency"])
    healthy = [r for r in levels if is_healthy(r)]
    if not healthy:
        return None, "所有档位均超出延迟/错误率阈值"

    peak = max(healthy, key=lambda r: r["fps"])
    target = peak["fps"] * (1 - fps_tolerance)
    best = next(r for r in healthy if r["fps"] >= target)

    note = (f"达到峰值吞吐 {peak['fps']:.2f} 页/s（C={peak['concurrency']}）的 "
            f"{1 - fps_tolerance:.0%}；再加并发只涨 <{fps_tolerance:.0%} 吞吐")
    higher = [r for r in levels if r["concurrency"] > peak["concurrency"]]
    if higher:
        first = higher[0]
        why = []
        if first["error_rate"] > error_rate_threshold:
            why.append(f"错误率 {first['error_rate']:.1%}")
        if first["e2e_p99_ms"] is None:
            why.append("无有效 p99")
        elif first["e2e_p99_ms"] > p99_threshold_ms:
            why.append(f"p99 {first['e2e_p99_ms'] / 1000:.0f}s 超阈值")
        if why:
            note += f"；C={first['concurrency']} 已不达标（{'、'.join(why)}）"
    elif peak is levels[-1]:
        note += "；梯度内吞吐仍在爬顶，实际上限可能更高"
    return best["concurrency"], note


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------

LEVEL_HEADER = (
    f"  {'C':>3s} {'FPS':>7s} {'req/s':>7s} {'块/页':>6s} "
    f"{'e2e p50':>9s} {'e2e p90':>9s} {'e2e p99':>9s} "
    f"{'out tok/s':>10s} {'错误':>7s} {'前缀缓存':>9s} {'显存MB':>8s}"
)


def fmt_ms(v):
    return f"{v:.0f}" if v is not None else "—"


def fmt_hit_rate(prefix_cache):
    if not prefix_cache:
        return "—"
    return f"{prefix_cache['hit_rate'] * 100:.0f}%"


def print_level(r):
    gpu = r.get("gpu") or {}
    print(
        f"  {r['concurrency']:>3d} {r['fps']:>7.3f} {r['req_per_s']:>7.2f} "
        f"{r['avg_blocks_per_page']:>6.1f} "
        f"{fmt_ms(r['e2e_p50_ms']):>9s} {fmt_ms(r['e2e_p90_ms']):>9s} "
        f"{fmt_ms(r['e2e_p99_ms']):>9s} "
        f"{r['completion_tokens_per_s']:>10.1f} "
        f"{r['error_rate'] * 100:>6.1f}% "
        f"{fmt_hit_rate(r.get('prefix_cache')):>9s} "
        f"{gpu.get('mem_used_mb_max', 0):>8.0f}"
    )




# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

async def bench_target(target, dataset, args, nonce_counter):
    adapter = ADAPTERS[target["adapter"]]

    print(f"\n{'=' * 100}")
    print(f"=== [{target['name']}] {target['base_url']}  model={target['model']}  GPU{target['gpu']}")

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        by_profile, failures = await preflight(
            session, target, adapter, dataset, args.max_tokens,
            args.timeout, args.block_cap,
        )

    levels = [int(x) for x in args.concurrency_levels.split(",") if x.strip()]
    profile_results = []
    all_blocks, all_pages = 0, 0

    for profile_name, _, _ in PROFILES:
        pages = by_profile.get(profile_name)
        if not pages:
            print(f"\n--- [{target['name']}] {profile_name}: 无可用页面，跳过 ---")
            continue
        all_pages += len(pages)
        all_blocks += sum(len(p.blocks) for p in pages)

        avg_blocks = sum(len(p.blocks) for p in pages) / len(pages)
        print(f"\n--- [{target['name']}] {profile_name}: "
              f"{len(pages)} 页样本, 平均 {avg_blocks:.1f} 块/页 ---")
        print(LEVEL_HEADER)

        level_results = []
        for c in levels:
            n_pages = max(args.pages_per_level, 2 * c)
            tasks = materialize_tasks(pages, n_pages, nonce_counter, args.cache_busting)
            r = await run_level(target, adapter.layout_prompt, tasks, c,
                                args.max_tokens, args.timeout,
                                args.block_concurrency, target.get("gpu"),
                                args.connection_limit)
            print_level(r)
            level_results.append(r)

        rec, note = pick_max_concurrency(
            level_results, args.p99_threshold_ms,
            args.error_rate_threshold, args.fps_tolerance,
        )
        print(f"  -> 建议最大支撑并发 = {rec}" + (f"  ({note})" if note else ""))
        profile_results.append({
            "profile": profile_name,
            "pages_in_pool": len(pages),
            "avg_blocks_per_page": avg_blocks,
            "levels": level_results,
            "recommended_concurrency": rec,
            "recommendation_note": note,
        })

    return {
        "name": target["name"], "base_url": target["base_url"],
        "model": target["model"], "gpu": target.get("gpu"),
        "adapter": target["adapter"],
        "avg_blocks_per_page": all_blocks / max(all_pages, 1),
        "preflight_failures": failures,
        "profiles": profile_results,
    }


async def run_bench(args):
    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"压测数据集: {args.dataset}  "
          f"({len(dataset['entries'])} 篇文档, "
          f"{sum(len(e['page_indices']) for e in dataset['entries'])} 页)")

    if not args.cache_busting:
        print("! 已关闭图片水印去重，各并发档会命中 vLLM 前缀缓存，吞吐读数偏乐观")

    targets = parse_targets(args.target) if args.target else DEFAULT_TARGETS
    nonce_counter = itertools.count(1)
    results = []
    for target in targets:
        results.append(await bench_target(target, dataset, args, nonce_counter))

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "concurrency_levels": args.concurrency_levels,
        "pages_per_level": args.pages_per_level,
        "max_tokens": args.max_tokens,
        "p99_threshold_ms": args.p99_threshold_ms,
        "error_rate_threshold": args.error_rate_threshold,
        "fps_tolerance": args.fps_tolerance,
        "cache_busting": args.cache_busting,
        "targets": results,
    }

    print_summary(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果: {args.output}")
    if args.html:
        write_html(report, args.html)
        print(f"HTML 报告: {args.html}")


def print_summary(report):
    print(f"\n{'=' * 100}\n=== 总结: 建议最大支撑并发 (并发文档数) ===")
    for t in report["targets"]:
        print(f"\n  [{t['name']}] GPU{t['gpu']}, 平均 {t['avg_blocks_per_page']:.1f} 块/页")
        for pr in t["profiles"]:
            lvl = next((l for l in pr["levels"]
                        if l["concurrency"] == pr["recommended_concurrency"]), None)
            fps = f"{lvl['fps']:.3f} 页/s" if lvl else "-"
            print(f"    {pr['profile']:<18s} 建议并发={str(pr['recommended_concurrency']):<4s} "
                  f"FPS={fps:<14s} {pr['recommendation_note']}")


def recompute_report(args):
    """用新阈值重算建议并发并重新出报告，不重跑压测。"""
    with open(args.input, encoding="utf-8") as f:
        report = json.load(f)

    report["p99_threshold_ms"] = args.p99_threshold_ms
    report["error_rate_threshold"] = args.error_rate_threshold
    report["fps_tolerance"] = args.fps_tolerance
    report.pop("fps_gain_threshold", None)   # 旧字段名，重算时清掉
    for t in report["targets"]:
        for pr in t["profiles"]:
            rec, note = pick_max_concurrency(
                pr["levels"], args.p99_threshold_ms,
                args.error_rate_threshold, args.fps_tolerance,
            )
            pr["recommended_concurrency"] = rec
            pr["recommendation_note"] = note

    print_summary(report)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {args.output}")
    if args.html:
        write_html(report, args.html)
        print(f"HTML 报告: {args.html}")


def parse_targets(raw):
    """格式: name=url,model,adapter[,gpu]"""
    targets = []
    for item in raw:
        name, rest = item.split("=", 1)
        fields = rest.split(",")
        targets.append({
            "name": name, "base_url": fields[0], "model": fields[1],
            "adapter": fields[2] if len(fields) > 2 else name,
            "gpu": int(fields[3]) if len(fields) > 3 else None,
        })
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="文档解析 VLM 端到端页级压测 (两阶段原生协议)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dataset", help="从 PDF 语料分层抽样构建压测数据集")
    b.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR)
    b.add_argument("--output", default=DEFAULT_DATASET)
    b.add_argument("--docs-per-profile", type=int, default=3, help="每档抽几篇文档，默认 3")
    b.add_argument("--pages-per-doc", type=int, default=3, help="每篇取几页(首/中/尾均匀)，默认 3")
    b.add_argument("--dpi", type=int, default=150, help="页面渲染 DPI，默认 150")
    b.add_argument("--seed", type=int, default=42, help="随机种子，默认 42 (可复现)")
    b.add_argument("--max-attempts", type=int, default=400,
                   help="每档最多尝试打开多少候选文件，默认 400")

    r = sub.add_parser("run", help="跑端到端页级压测")
    r.add_argument("--dataset", default=DEFAULT_DATASET)
    r.add_argument("--target", action="append",
                   help="name=url,model,adapter[,gpu]，可重复；不指定则用默认两个服务")
    r.add_argument("--concurrency-levels", default="1,2,4,8,16",
                   help="并发文档数梯度，默认 1,2,4,8,16")
    r.add_argument("--pages-per-level", type=int, default=18,
                   help="每个并发档跑多少个页面任务(不足 2*C 时取 2*C)，默认 18")
    r.add_argument("--max-tokens", type=int, default=2048)
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--block-concurrency", type=int, default=0,
                   help="单页内 block 请求的并发上限，0=不限(与官方客户端一致)")
    r.add_argument("--block-cap", type=int, default=0,
                   help="单页最多取多少个 block，0=不限。仅用于缩短调试耗时")
    r.add_argument("--connection-limit", type=int, default=256,
                   help="客户端 HTTP 连接池上限，默认 256。0=不限，但高并发下会因"
                        "瞬时开出上千条连接触发 Connection reset，测到的是连接层而非推理算力")
    r.add_argument("--no-cache-busting", dest="cache_busting", action="store_false",
                   help="关闭图片唯一水印。默认开启，使每个请求必然 miss vLLM 前缀缓存，"
                        "对齐'生产里每页都是新的'；关闭后可对比缓存全命中时的上限吞吐")
    r.set_defaults(cache_busting=True)
    r.add_argument("--p99-threshold-ms", type=float, default=30000.0,
                   help="判定健康档的单页 e2e p99 阈值(ms)，默认 30000")
    r.add_argument("--error-rate-threshold", type=float, default=0.01)
    r.add_argument("--fps-tolerance", type=float, default=0.05,
                   help="允许偏离峰值吞吐的比例，默认 0.05：建议并发取最小的、"
                        "吞吐已达峰值 95%% 的达标档位")
    r.add_argument("--output", default="ocr_e2e_report.json")
    r.add_argument("--html", default="ocr_e2e_report.html")

    rp = sub.add_parser("report", help="用新阈值重算建议并发并重出报告，不重跑压测")
    rp.add_argument("--input", default="ocr_e2e_report.json", help="已有的结果 JSON")
    rp.add_argument("--output", default="ocr_e2e_report.json")
    rp.add_argument("--html", default="ocr_e2e_report.html")
    rp.add_argument("--p99-threshold-ms", type=float, default=30000.0)
    rp.add_argument("--error-rate-threshold", type=float, default=0.01)
    rp.add_argument("--fps-tolerance", type=float, default=0.05)

    args = parser.parse_args()

    if args.cmd == "report":
        recompute_report(args)
    elif args.cmd == "build-dataset":
        dataset = build_dataset(args.pdf_dir, args.docs_per_profile, args.pages_per_doc,
                                args.dpi, args.seed, args.max_attempts)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        n_pages = sum(len(e["page_indices"]) for e in dataset["entries"])
        print(f"\n数据集已写入 {args.output}: "
              f"{len(dataset['entries'])} 篇文档 / {n_pages} 页")
    else:
        asyncio.run(run_bench(args))


if __name__ == "__main__":
    main()
