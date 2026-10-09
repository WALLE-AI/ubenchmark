#!/usr/bin/env python3
"""验证 MinerU2.5-Pro / TeleOCR 的原生两阶段 prompt 在已部署 vLLM 服务上的行为。

这是 bench_ocr_e2e.py 的前置检查：确认 layout 阶段返回的是版面框而不是散文，
并确认 block 阶段能对裁剪出的版面块返回真实 OCR 文本。

用法:
  /home/dataset-s3-0/gaojing/anaconda3/envs/docparserEnv/bin/python probe_native_prompt.py
"""

import base64
import io
import json
import re
import urllib.request

import fitz
from PIL import Image

PDF = "/home/dataset-s3-0/datasets/PDFs/data/GBT51121-2015 风力发电工程施工与验收规范.pdf"
PAGE_INDEX = 20
LAYOUT_SIZE = (1036, 1036)

MINERU_LAYOUT_RE = re.compile(
    r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"
)
TELEOCR_LAYOUT_RE = re.compile(r"^<box:([\d\s]+)><label:(\w+)><([^>]+)>$")

TARGETS = [
    {
        "name": "mineru",
        "url": "http://127.0.0.1:8003",
        "model": "mineru2.5-pro-1.2b",
        "layout_prompt": "\nLayout Detection:",
        "text_prompt": "\nText Recognition:",
        "layout_re": MINERU_LAYOUT_RE,
    },
    {
        "name": "teleocr",
        "url": "http://127.0.0.1:8004",
        "model": "teleocr",
        "layout_prompt": "\nAnalyze the image layout.",
        "text_prompt": "\nPlease output the text content from the image.",
        "layout_re": TELEOCR_LAYOUT_RE,
    },
]


def data_uri(image):
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def call(base_url, model, prompt, image, max_tokens=2048):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.01,
        "top_k": 1,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        # MinerU 的 layout 输出用 <|box_start|>/<|ref_start|> 等特殊 token 分隔字段，
        # vLLM OpenAI 接口默认 skip_special_tokens=True 会把它们剥掉，导致无法解析。
        "skip_special_tokens": False,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri(image)}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    choice = resp["choices"][0]
    return choice["message"]["content"], resp["usage"], choice["finish_reason"]


def mineru_boxes(text):
    """返回 [(type, (x1,y1,x2,y2) 归一化)]，坐标 0-1000 -> 0-1。"""
    out = []
    for line in text.split("\n"):
        m = MINERU_LAYOUT_RE.match(line.strip())
        if not m:
            continue
        x1, y1, x2, y2 = (int(v) for v in m.groups()[:4])
        if any(c < 0 or c > 1000 for c in (x1, y1, x2, y2)):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x1 == x2 or y1 == y2:
            continue
        out.append((m.group(5).lower(), (x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000)))
    return out


def teleocr_boxes(text):
    """返回 [(type, (x1,y1,x2,y2) 归一化)]，多点多边形取外接矩形。"""
    out = []
    for line in text.split("\n"):
        m = TELEOCR_LAYOUT_RE.match(line.strip())
        if not m:
            continue
        nums = [int(v) for v in m.group(1).split()]
        if len(nums) < 4 or len(nums) % 2:
            continue
        xs, ys = nums[0::2], nums[1::2]
        if any(c < 0 or c > 1000 for c in nums):
            continue
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        if x1 == x2 or y1 == y2:
            continue
        out.append((m.group(2).lower(), (x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000)))
    return out


def main():
    doc = fitz.open(PDF)
    pix = doc.load_page(PAGE_INDEX).get_pixmap(dpi=150)
    page = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    print(f"原页尺寸: {page.width}x{page.height}px  (dpi=150, page={PAGE_INDEX})")

    layout_image = page.resize(LAYOUT_SIZE, Image.Resampling.BICUBIC)

    for t in TARGETS:
        print(f"\n{'='*70}\n##### {t['name']}  layout 阶段")
        try:
            text, usage, finish = call(t["url"], t["model"], t["layout_prompt"], layout_image)
        except Exception as e:  # noqa: BLE001
            print(f"  请求失败: {e}")
            continue

        lines = [l for l in text.split("\n") if l.strip()]
        boxes = mineru_boxes(text) if t["name"] == "mineru" else teleocr_boxes(text)
        matched = sum(1 for l in lines if t["layout_re"].match(l.strip()))
        print(f"  in={usage['prompt_tokens']} out={usage['completion_tokens']} finish={finish}")
        print(f"  输出行数={len(lines)}  正则匹配={matched}  解析出 block={len(boxes)}")
        for l in lines[:5]:
            print(f"    {l[:110]}")
        if not boxes:
            print("  !! 未解析出任何 block —— prompt 或图片预处理需要调整")
            continue

        types = {}
        for bt, _ in boxes:
            types[bt] = types.get(bt, 0) + 1
        print(f"  block 类型分布: {types}")

        # block 阶段：取面积最大的 text 类块做一次识别
        text_blocks = [b for b in boxes if b[0] in ("text", "title")] or boxes
        text_blocks.sort(key=lambda b: (b[1][2] - b[1][0]) * (b[1][3] - b[1][1]), reverse=True)
        bt, (nx1, ny1, nx2, ny2) = text_blocks[0]
        crop = page.crop((nx1 * page.width, ny1 * page.height,
                          nx2 * page.width, ny2 * page.height))
        print(f"\n##### {t['name']}  block 阶段 (type={bt}, crop={crop.width}x{crop.height}px)")
        try:
            btext, busage, bfinish = call(t["url"], t["model"], t["text_prompt"], crop)
            print(f"  in={busage['prompt_tokens']} out={busage['completion_tokens']} finish={bfinish}")
            print(f"  识别结果: {btext[:300]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  请求失败: {e}")


if __name__ == "__main__":
    main()
