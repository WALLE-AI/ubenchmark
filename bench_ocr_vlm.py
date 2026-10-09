#!/usr/bin/env python3
"""
压测 vLLM OpenAI 兼容多模态文档解析服务 (/v1/chat/completions, image_url + 文本)。
面向 MinerU2.5-Pro-2605-1.2B / TeleOCR 这类文档解析 VLM，覆盖三个维度:

  1. 文档尺寸/分辨率支撑能力: 不同分辨率的合成"文档图片" (small/a4_150dpi/a4_300dpi/large)
     在固定并发下的延迟、吞吐、错误率。
  2. 单请求多页支撑能力: 单次请求内塞入 N 张图片(模拟多页 PDF)，从小到大探测,
     找到开始报错(通常是超出 max-model-len 或 OOM)前的最大页数。
  3. 并发数量 vs 性能: 对每个文档尺寸做并发梯度扫描，统计 req/s、tokens/s、
     延迟 P50/P90/P99、错误率，并给出建议的最大支撑并发。
  4. (可选) 首 token 延迟 (TTFT): 用流式请求单独采样，低并发下测量。

用法:
  # 用默认配置直接跑两个已部署的模型 (mineru: 8003, teleocr: 8004)
  python3 bench_ocr_vlm.py

  # 自定义
  python3 bench_ocr_vlm.py \
    --target mineru=http://127.0.0.1:8003,mineru2.5-pro-1.2b \
    --target teleocr=http://127.0.0.1:8004,teleocr \
    --doc-sizes a4_150dpi,a4_300dpi \
    --concurrency-levels 1,2,4,8,16 \
    --requests-per-level 20 \
    --pages-sweep 1,2,4,8,16,32 \
    --output ocr_bench_report.json
"""

import argparse
import asyncio
import base64
import io
import json
import random
import statistics
import time

import aiohttp
from PIL import Image, ImageDraw, ImageFont

DEFAULT_TARGETS = [
    ("mineru", "http://127.0.0.1:8003", "mineru2.5-pro-1.2b"),
    ("teleocr", "http://127.0.0.1:8004", "teleocr"),
]

# 常见文档扫描分辨率预设 (宽, 高) px
DOC_SIZE_PRESETS = {
    "small": (800, 600),          # 低分辨率照片/缩略图
    "a4_150dpi": (1240, 1754),    # A4 @ 150dpi，常见扫描件
    "a4_300dpi": (2480, 3508),    # A4 @ 300dpi，高清扫描件
    "large": (3500, 4900),        # 大幅面/高清拍摄
}

OCR_PROMPT = "请识别并输出这张文档图片中的全部文字内容，保持原始阅读顺序与版式，输出 Markdown。"

DEJAVU_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]

LOREM_WORDS = (
    "document parsing layout table formula reading order markdown benchmark "
    "throughput latency concurrency vision language model inference token "
    "invoice report contract page header footer paragraph column figure chart"
).split()


def _load_font(size):
    for path in DEJAVU_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _random_paragraph(n_words):
    return " ".join(random.choice(LOREM_WORDS) for _ in range(n_words))


def make_document_image(width, height, seed):
    """生成一张模拟文档扫描件的合成图片 (标题 + 多段落文字 + 一个表格)。"""
    rng = random.Random(seed)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    margin = int(width * 0.06)
    title_font = _load_font(max(18, int(height * 0.03)))
    body_font = _load_font(max(12, int(height * 0.016)))

    y = margin
    draw.text((margin, y), f"Synthetic Document Report #{seed}", font=title_font, fill="black")
    y += int(height * 0.05)

    line_h = int(height * 0.022)
    max_line_width = width - 2 * margin
    while y < height * 0.6:
        line = _random_paragraph(rng.randint(8, 14))
        draw.text((margin, y), line, font=body_font, fill="black")
        y += line_h
        if rng.random() < 0.15:
            y += line_h // 2  # 段落间距

    # 简单表格，制造版式复杂度
    table_top = int(height * 0.62)
    table_bottom = int(height * 0.92)
    cols = 4
    rows = 6
    col_w = (width - 2 * margin) / cols
    row_h = (table_bottom - table_top) / rows
    for r in range(rows + 1):
        y_line = table_top + r * row_h
        draw.line([(margin, y_line), (width - margin, y_line)], fill="black", width=1)
    for c in range(cols + 1):
        x_line = margin + c * col_w
        draw.line([(x_line, table_top), (x_line, table_bottom)], fill="black", width=1)
    for r in range(rows):
        for c in range(cols):
            cell_text = rng.choice(LOREM_WORDS)
            draw.text(
                (margin + c * col_w + 4, table_top + r * row_h + 4),
                cell_text, font=body_font, fill="black",
            )

    return img


def image_to_data_uri(img, fmt="JPEG", quality=90):
    buf = io.BytesIO()
    save_fmt = "JPEG" if fmt.upper() in ("JPG", "JPEG") else "PNG"
    if save_fmt == "JPEG":
        img.save(buf, format=save_fmt, quality=quality)
        mime = "image/jpeg"
    else:
        img.save(buf, format=save_fmt)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}", len(buf.getvalue())


class ImageCache:
    """按 (preset, index) 缓存生成的图片 data URI，避免重复生成/编码开销污染压测结果。"""

    def __init__(self, image_format, jpeg_quality):
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self._cache = {}

    def get(self, preset, variant=0):
        key = (preset, variant)
        if key not in self._cache:
            w, h = DOC_SIZE_PRESETS[preset]
            img = make_document_image(w, h, seed=hash(key) % 100000)
            data_uri, nbytes = image_to_data_uri(img, self.image_format, self.jpeg_quality)
            self._cache[key] = (data_uri, nbytes, w, h)
        return self._cache[key]


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def build_payload(model, data_uris, max_tokens, stream=False):
    content = [{"type": "text", "text": OCR_PROMPT}]
    for uri in data_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }


async def send_one(session, url, payload, timeout_s):
    t0 = time.perf_counter()
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as resp:
            body = await resp.read()
            dt = time.perf_counter() - t0
            if resp.status != 200:
                return dt, False, f"HTTP {resp.status}: {body[:300]!r}", None, None
            data = json.loads(body)
            if "choices" not in data or not data["choices"]:
                return dt, False, f"malformed response: {body[:300]!r}", None, None
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            return dt, True, None, prompt_tokens, completion_tokens
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t0
        return dt, False, f"{type(e).__name__}: {e}", None, None


async def send_one_stream(session, url, payload, timeout_s):
    """流式请求，返回 (ttft_s, total_s, ok, err)。"""
    t0 = time.perf_counter()
    ttft = None
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as resp:
            if resp.status != 200:
                body = await resp.read()
                return None, time.perf_counter() - t0, False, f"HTTP {resp.status}: {body[:300]!r}"
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
            total = time.perf_counter() - t0
            return ttft, total, True, None
    except Exception as e:  # noqa: BLE001
        return ttft, time.perf_counter() - t0, False, f"{type(e).__name__}: {e}"


async def run_level(base_url, model, concurrency, total_requests, data_uris, max_tokens, timeout_s):
    url = f"{base_url}/v1/chat/completions"
    payload = build_payload(model, data_uris, max_tokens)

    sem = asyncio.Semaphore(concurrency)
    latencies = []
    errors = []
    prompt_tok_sum = 0
    completion_tok_sum = 0

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:

        async def worker():
            nonlocal prompt_tok_sum, completion_tok_sum
            async with sem:
                dt, ok, err, ptok, ctok = await send_one(session, url, payload, timeout_s)
                latencies.append(dt)
                if not ok:
                    errors.append(err)
                else:
                    prompt_tok_sum += ptok or 0
                    completion_tok_sum += ctok or 0

        wall_t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(total_requests)))
        wall_dt = time.perf_counter() - wall_t0

    n_ok = total_requests - len(errors)
    req_per_s = n_ok / wall_dt if wall_dt > 0 else 0.0
    out_tok_per_s = completion_tok_sum / wall_dt if wall_dt > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "ok": n_ok,
        "errors": len(errors),
        "error_samples": errors[:3],
        "wall_time_s": wall_dt,
        "req_per_s": req_per_s,
        "output_tokens_per_s": out_tok_per_s,
        "avg_prompt_tokens": (prompt_tok_sum / n_ok) if n_ok else 0.0,
        "avg_completion_tokens": (completion_tok_sum / n_ok) if n_ok else 0.0,
        "latency_avg_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
        "latency_p50_ms": percentile(latencies, 50) * 1000,
        "latency_p90_ms": percentile(latencies, 90) * 1000,
        "latency_p99_ms": percentile(latencies, 99) * 1000,
        "latency_max_ms": max(latencies) * 1000 if latencies else 0.0,
    }


def print_level_result(r):
    err_rate = (r["errors"] / r["total_requests"] * 100) if r["total_requests"] else 0
    print(
        f"  C={r['concurrency']:<4d} "
        f"req/s={r['req_per_s']:6.2f}  out_tok/s={r['output_tokens_per_s']:7.1f}  "
        f"avg_in_tok={r['avg_prompt_tokens']:6.0f} avg_out_tok={r['avg_completion_tokens']:6.0f}  "
        f"p50={r['latency_p50_ms']:8.1f}ms p90={r['latency_p90_ms']:8.1f}ms "
        f"p99={r['latency_p99_ms']:8.1f}ms max={r['latency_max_ms']:9.1f}ms  "
        f"err={r['errors']}/{r['total_requests']} ({err_rate:.1f}%)"
    )
    if r["error_samples"]:
        for e in r["error_samples"]:
            print(f"      ! {e}")


def pick_max_supported(results, p99_threshold_ms, error_rate_threshold):
    healthy = []
    for r in results:
        err_rate = (r["errors"] / r["total_requests"]) if r["total_requests"] else 1.0
        if err_rate <= error_rate_threshold and r["latency_p99_ms"] <= p99_threshold_ms:
            healthy.append(r)
    if not healthy:
        return None
    return max(healthy, key=lambda r: r["concurrency"])


async def warmup(base_url, model, data_uris, max_tokens, timeout_s):
    url = f"{base_url}/v1/chat/completions"
    payload = build_payload(model, data_uris, max_tokens)
    async with aiohttp.ClientSession() as session:
        await send_one(session, url, payload, timeout_s)


async def bench_doc_size(name, base_url, model, preset, cache, levels, requests_per_level,
                          max_tokens, timeout_s, p99_threshold_ms, error_rate_threshold):
    data_uri, nbytes, w, h = cache.get(preset)
    print(f"\n--- [{name}] 文档尺寸={preset} ({w}x{h}px, {nbytes/1024:.0f}KB) ---")

    print("  预热中...")
    try:
        await warmup(base_url, model, [data_uri], max_tokens, timeout_s)
    except Exception as e:  # noqa: BLE001
        print(f"  预热失败: {e}")

    results = []
    for c in levels:
        n = max(requests_per_level, c)
        r = await run_level(base_url, model, c, n, [data_uri], max_tokens, timeout_s)
        print_level_result(r)
        results.append(r)

    best = pick_max_supported(results, p99_threshold_ms, error_rate_threshold)
    if best:
        print(
            f"  建议最大支撑并发 ≈ {best['concurrency']} "
            f"(p99={best['latency_p99_ms']:.1f}ms, req/s={best['req_per_s']:.2f}, "
            f"out_tok/s={best['output_tokens_per_s']:.1f})"
        )
    else:
        print("  未找到满足阈值的健康并发档位，请降低并发或检查服务日志。")

    return {
        "doc_size": preset, "width": w, "height": h, "image_bytes": nbytes,
        "results": results,
        "recommended_concurrency": best["concurrency"] if best else None,
    }


async def bench_pages_capacity(name, base_url, model, preset, cache, pages_sweep, max_tokens, timeout_s):
    """单请求内塞入 N 张图片，找最大可支撑页数 (串行、单并发，避免和显存压力混淆)。"""
    print(f"\n--- [{name}] 单请求多页支撑能力探测 (doc_size={preset}) ---")
    url = f"{base_url}/v1/chat/completions"
    max_ok_pages = 0
    detail = []
    async with aiohttp.ClientSession() as session:
        for n_pages in pages_sweep:
            data_uris = [cache.get(preset, variant=i)[0] for i in range(n_pages)]
            payload = build_payload(model, data_uris, max_tokens)
            dt, ok, err, ptok, ctok = await send_one(session, url, payload, timeout_s)
            status = "OK" if ok else "FAIL"
            print(f"  pages={n_pages:<3d} {status}  耗时={dt*1000:.0f}ms  in_tok={ptok}  err={err or ''}")
            detail.append({"pages": n_pages, "ok": ok, "latency_ms": dt * 1000,
                            "prompt_tokens": ptok, "error": err})
            if ok:
                max_ok_pages = n_pages
            else:
                break
    print(f"  --- 最大支撑页数 ≈ {max_ok_pages} ---")
    return {"doc_size": preset, "max_supported_pages": max_ok_pages, "detail": detail}


async def bench_ttft(name, base_url, model, preset, cache, n_samples, max_tokens, timeout_s):
    print(f"\n--- [{name}] 首 token 延迟(TTFT) 采样 (doc_size={preset}, n={n_samples}, 串行) ---")
    url = f"{base_url}/v1/chat/completions"
    data_uri, _, _, _ = cache.get(preset)
    payload = build_payload(model, [data_uri], max_tokens, stream=True)
    ttfts, totals, errors = [], [], []
    async with aiohttp.ClientSession() as session:
        for i in range(n_samples):
            ttft, total, ok, err = await send_one_stream(session, url, payload, timeout_s)
            if ok and ttft is not None:
                ttfts.append(ttft)
                totals.append(total)
            else:
                errors.append(err)
    if ttfts:
        print(
            f"  TTFT avg={statistics.mean(ttfts)*1000:.0f}ms p50={percentile(ttfts,50)*1000:.0f}ms "
            f"p90={percentile(ttfts,90)*1000:.0f}ms  总耗时avg={statistics.mean(totals)*1000:.0f}ms  "
            f"错误={len(errors)}/{n_samples}"
        )
    else:
        print(f"  全部失败: {errors[:3]}")
    return {
        "doc_size": preset, "n_samples": n_samples, "errors": len(errors),
        "ttft_avg_ms": statistics.mean(ttfts) * 1000 if ttfts else None,
        "ttft_p50_ms": percentile(ttfts, 50) * 1000 if ttfts else None,
        "ttft_p90_ms": percentile(ttfts, 90) * 1000 if ttfts else None,
    }


async def bench_target(name, base_url, model, args, cache):
    print(f"\n=== [{name}] {base_url}  model={model} ===")
    levels = [int(x) for x in args.concurrency_levels.split(",") if x.strip()]
    doc_sizes = [s.strip() for s in args.doc_sizes.split(",") if s.strip()]

    size_results = []
    for preset in doc_sizes:
        r = await bench_doc_size(
            name, base_url, model, preset, cache, levels, args.requests_per_level,
            args.max_tokens, args.timeout, args.p99_threshold_ms, args.error_rate_threshold,
        )
        size_results.append(r)

    pages_result = None
    if args.pages_sweep:
        pages_sweep = [int(x) for x in args.pages_sweep.split(",") if x.strip()]
        pages_result = await bench_pages_capacity(
            name, base_url, model, args.pages_doc_size, cache, pages_sweep,
            args.max_tokens, args.timeout,
        )

    ttft_result = None
    if args.measure_ttft:
        ttft_result = await bench_ttft(
            name, base_url, model, doc_sizes[0], cache, args.ttft_samples,
            args.max_tokens, args.timeout,
        )

    return {
        "name": name, "base_url": base_url, "model": model,
        "doc_size_results": size_results,
        "pages_capacity": pages_result,
        "ttft": ttft_result,
    }


async def main_async(args):
    targets = parse_targets(args.target) if args.target else DEFAULT_TARGETS
    cache = ImageCache(args.image_format, args.jpeg_quality)

    summary = []
    for name, url, model in targets:
        res = await bench_target(name, url, model, args, cache)
        summary.append(res)

    print("\n=== 总结 ===")
    for s in summary:
        print(f"  [{s['name']}]")
        for sr in s["doc_size_results"]:
            print(f"    {sr['doc_size']}: 建议最大支撑并发 = {sr['recommended_concurrency']}")
        if s["pages_capacity"]:
            print(f"    单请求最大支撑页数 ≈ {s['pages_capacity']['max_supported_pages']}")
        if s["ttft"] and s["ttft"]["ttft_avg_ms"] is not None:
            print(f"    TTFT avg = {s['ttft']['ttft_avg_ms']:.0f}ms")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已写入: {args.output}")


def parse_targets(raw_targets):
    targets = []
    for item in raw_targets:
        name, rest = item.split("=", 1)
        url, model = rest.split(",", 1)
        targets.append((name.strip(), url.strip().rstrip("/"), model.strip()))
    return targets


def main():
    parser = argparse.ArgumentParser(description="压测文档解析 VLM (MinerU2.5-Pro / TeleOCR) vLLM 服务")
    parser.add_argument("--target", action="append",
                         help="格式: name=http://host:port,model_name，可重复指定，不指定则用默认两个服务")
    parser.add_argument("--doc-sizes", default="a4_150dpi",
                         help=f"逗号分隔，可选: {','.join(DOC_SIZE_PRESETS.keys())}，默认 a4_150dpi")
    parser.add_argument("--concurrency-levels", default="1,2,4,8,16",
                         help="逗号分隔的并发梯度，默认 1,2,4,8,16")
    parser.add_argument("--requests-per-level", type=int, default=20,
                         help="每个并发档发送的请求总数（不足并发数时取并发数），默认 20")
    parser.add_argument("--max-tokens", type=int, default=2048, help="每次请求的输出 token 上限，默认 2048")
    parser.add_argument("--timeout", type=float, default=180.0, help="单请求超时(s)，默认 180")
    parser.add_argument("--p99-threshold-ms", type=float, default=15000.0,
                         help="判定'健康'档位的 p99 延迟阈值(ms)，默认 15000")
    parser.add_argument("--error-rate-threshold", type=float, default=0.01,
                         help="判定'健康'档位的最大错误率，默认 0.01 (1%%)")
    parser.add_argument("--pages-sweep", default="1,2,4,8,16,32",
                         help="单请求多页支撑探测的页数梯度，逗号分隔；置空字符串可跳过该测试")
    parser.add_argument("--pages-doc-size", default="a4_150dpi",
                         help="多页支撑探测使用的文档尺寸预设，默认 a4_150dpi")
    parser.add_argument("--measure-ttft", action="store_true", default=True,
                         help="是否测量首 token 延迟 (流式请求，低并发串行采样)，默认开启")
    parser.add_argument("--no-ttft", dest="measure_ttft", action="store_false",
                         help="关闭 TTFT 测量")
    parser.add_argument("--ttft-samples", type=int, default=5, help="TTFT 采样请求数，默认 5")
    parser.add_argument("--image-format", choices=["jpeg", "png"], default="jpeg",
                         help="合成文档图片编码格式，默认 jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG 质量，默认 90")
    parser.add_argument("--output", default=None, help="将详细结果写入 JSON 文件路径")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
