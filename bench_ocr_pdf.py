#!/usr/bin/env python3
"""
用真实 PDF 语料 (/home/dataset-s3-0/datasets/PDFs/data) 压测文档解析 VLM
(MinerU2.5-Pro-2605-1.2B / TeleOCR, vLLM OpenAI 兼容 /v1/chat/completions)。

相比 bench_ocr_vlm.py (合成图片)，这里从真实语料里按 "文件大小 x 页数(章节数代理)"
两个维度挑选代表性文档，更贴近生产场景:

  1. 扫描语料目录，按文件大小分档 (tiny/small/medium/large/xlarge)。
  2. 对每个大小档，随机抽样打开文件 (PyMuPDF)，找到页数落在对应"篇幅"档位
     (short/medium/long/verylong) 的代表文档，同时记录书签(TOC)层级数作为
     "章节数"的参考信息。
  3. 对每个 (大小, 篇幅) 代表文档:
       a. 用其第 1 页渲染成图片，做并发梯度压测 (吞吐/延迟/tokens/s)。
       b. 用其真实连续页面，做"单请求多页"容量探测 (模拟整章/整份传入)。
       c. (可选) 流式请求采样首 token 延迟 (TTFT)。

依赖: PyMuPDF (fitz)，请用 docparserEnv 环境运行:
  /home/dataset-s3-0/gaojing/anaconda3/envs/docparserEnv/bin/python bench_ocr_pdf.py

用法:
  python3 bench_ocr_pdf.py \
    --pdf-dir /home/dataset-s3-0/datasets/PDFs/data \
    --concurrency-levels 1,2,4,8,16 \
    --requests-per-level 20 \
    --output ocr_pdf_bench_report.json
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import sys

import aiohttp
import fitz  # PyMuPDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ocr_vlm import (  # noqa: E402
    DEFAULT_TARGETS,
    build_payload,
    parse_targets,
    percentile,
    pick_max_supported,
    print_level_result,
    run_level,
    send_one,
    send_one_stream,
)

DEFAULT_PDF_DIR = "/home/dataset-s3-0/datasets/PDFs/data"

SIZE_TIERS = [
    ("tiny", 0, 200 * 1024),
    ("small", 200 * 1024, 2 * 1024 * 1024),
    ("medium", 2 * 1024 * 1024, 10 * 1024 * 1024),
    ("large", 10 * 1024 * 1024, 50 * 1024 * 1024),
    ("xlarge", 50 * 1024 * 1024, float("inf")),
]

PAGE_TIERS = [
    ("short", 1, 5),
    ("medium", 6, 30),
    ("long", 31, 100),
    ("verylong", 101, float("inf")),
]

# (size_tier, page_tier) 组合，覆盖 "大小 x 篇幅(章节数代理)" 的代表性场景
PROFILES = [
    ("tiny_short", "tiny", "short"),
    ("small_medium", "small", "medium"),
    ("medium_long", "medium", "long"),
    ("large_long", "large", "long"),
    ("xlarge_verylong", "xlarge", "verylong"),
]


def scan_catalog(pdf_dir):
    catalog = []
    for name in os.listdir(pdf_dir):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, name)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        catalog.append((path, size))
    return catalog


def size_tier_of(size):
    for name, lo, hi in SIZE_TIERS:
        if lo <= size < hi:
            return name
    return SIZE_TIERS[-1][0]


def page_tier_of(n_pages):
    for name, lo, hi in PAGE_TIERS:
        if lo <= n_pages <= hi:
            return name
    return PAGE_TIERS[-1][0]


def pick_representatives(catalog, max_attempts_per_profile=60, seed=42):
    """对每个 (size_tier, page_tier) profile，随机尝试打开候选文件，找到匹配页数档位的代表文档。"""
    rng = random.Random(seed)
    by_size_tier = {}
    for path, size in catalog:
        by_size_tier.setdefault(size_tier_of(size), []).append((path, size))

    reps = {}
    for profile_name, size_tier, page_tier in PROFILES:
        candidates = list(by_size_tier.get(size_tier, []))
        rng.shuffle(candidates)
        found = None
        for path, size in candidates[:max_attempts_per_profile]:
            try:
                doc = fitz.open(path)
                if doc.needs_pass:
                    doc.close()
                    continue
                n_pages = len(doc)
                if page_tier_of(n_pages) != page_tier:
                    doc.close()
                    continue
                try:
                    toc = doc.get_toc()
                    n_chapters = sum(1 for e in toc if e[0] == 1) or None
                except Exception:  # noqa: BLE001
                    n_chapters = None
                doc.close()
                found = {
                    "path": path, "size": size, "pages": n_pages,
                    "chapters": n_chapters,
                }
                break
            except Exception:  # noqa: BLE001
                continue
        if found:
            reps[profile_name] = found
    return reps


def render_page(path, page_index, dpi):
    doc = fitz.open(path)
    try:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("jpeg"), pix.width, pix.height
    finally:
        doc.close()


class PdfPageCache:
    def __init__(self, dpi):
        self.dpi = dpi
        self._cache = {}

    def get(self, path, page_index):
        key = (path, page_index)
        if key not in self._cache:
            jpg_bytes, w, h = render_page(path, page_index, self.dpi)
            import base64
            b64 = base64.b64encode(jpg_bytes).decode("ascii")
            data_uri = f"data:image/jpeg;base64,{b64}"
            self._cache[key] = (data_uri, len(jpg_bytes), w, h)
        return self._cache[key]


async def warmup(base_url, model, data_uris, max_tokens, timeout_s):
    url = f"{base_url}/v1/chat/completions"
    payload = build_payload(model, data_uris, max_tokens)
    async with aiohttp.ClientSession() as session:
        await send_one(session, url, payload, timeout_s)


async def bench_profile_throughput(name, base_url, model, profile_name, rep, cache,
                                    levels, requests_per_level, max_tokens, timeout_s,
                                    p99_threshold_ms, error_rate_threshold):
    data_uri, nbytes, w, h = cache.get(rep["path"], 0)
    fname = os.path.basename(rep["path"])
    print(
        f"\n--- [{name}] profile={profile_name}  file='{fname[:50]}'  "
        f"size={rep['size']/1024/1024:.2f}MB  pages={rep['pages']}  chapters={rep['chapters']}  "
        f"render={w}x{h}px({nbytes/1024:.0f}KB) ---"
    )

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
        print(f"  建议最大支撑并发 ≈ {best['concurrency']} (p99={best['latency_p99_ms']:.1f}ms)")
    else:
        print("  未找到满足阈值的健康并发档位。")

    return {
        "profile": profile_name, "file": fname, "size_bytes": rep["size"],
        "pages": rep["pages"], "chapters": rep["chapters"],
        "render_width": w, "render_height": h, "render_bytes": nbytes,
        "results": results,
        "recommended_concurrency": best["concurrency"] if best else None,
    }


async def bench_profile_multipage(name, base_url, model, profile_name, rep, cache,
                                   pages_sweep, max_pages_cap, max_tokens, timeout_s):
    n_pages_avail = rep["pages"]
    sweep = [p for p in pages_sweep if p <= min(n_pages_avail, max_pages_cap)]
    if not sweep:
        sweep = [1]
    print(f"\n--- [{name}] profile={profile_name} 单请求多页(真实连续页)容量探测，可用页数={n_pages_avail} ---")

    url = f"{base_url}/v1/chat/completions"
    max_ok_pages = 0
    detail = []
    async with aiohttp.ClientSession() as session:
        for n in sweep:
            try:
                data_uris = [cache.get(rep["path"], i)[0] for i in range(n)]
            except Exception as e:  # noqa: BLE001
                print(f"  pages={n:<3d} 渲染失败: {e}")
                break
            payload = build_payload(model, data_uris, max_tokens)
            dt, ok, err, ptok, ctok = await send_one(session, url, payload, timeout_s)
            status = "OK" if ok else "FAIL"
            print(f"  pages={n:<3d} {status}  耗时={dt*1000:.0f}ms  in_tok={ptok}  err={err or ''}")
            detail.append({"pages": n, "ok": ok, "latency_ms": dt * 1000,
                            "prompt_tokens": ptok, "error": err})
            if ok:
                max_ok_pages = n
            else:
                break
    print(f"  --- profile={profile_name} 最大支撑页数 ≈ {max_ok_pages} (语料实际页数={n_pages_avail}) ---")
    return {"profile": profile_name, "pages_available": n_pages_avail,
            "max_supported_pages": max_ok_pages, "detail": detail}


async def bench_profile_ttft(name, base_url, model, profile_name, rep, cache, n_samples, max_tokens, timeout_s):
    data_uri, _, _, _ = cache.get(rep["path"], 0)
    print(f"\n--- [{name}] profile={profile_name} TTFT 采样 (n={n_samples}, 串行) ---")
    url = f"{base_url}/v1/chat/completions"
    payload = build_payload(model, [data_uri], max_tokens, stream=True)
    ttfts, totals, errors = [], [], []
    async with aiohttp.ClientSession() as session:
        for _ in range(n_samples):
            ttft, total, ok, err = await send_one_stream(session, url, payload, timeout_s)
            if ok and ttft is not None:
                ttfts.append(ttft)
                totals.append(total)
            else:
                errors.append(err)
    if ttfts:
        print(
            f"  TTFT avg={statistics.mean(ttfts)*1000:.0f}ms p50={percentile(ttfts,50)*1000:.0f}ms  "
            f"总耗时avg={statistics.mean(totals)*1000:.0f}ms  错误={len(errors)}/{n_samples}"
        )
    else:
        print(f"  全部失败: {errors[:3]}")
    return {
        "profile": profile_name, "n_samples": n_samples, "errors": len(errors),
        "ttft_avg_ms": statistics.mean(ttfts) * 1000 if ttfts else None,
        "ttft_p50_ms": percentile(ttfts, 50) * 1000 if ttfts else None,
    }


async def bench_target(name, base_url, model, reps, args):
    print(f"\n=== [{name}] {base_url}  model={model} ===")
    levels = [int(x) for x in args.concurrency_levels.split(",") if x.strip()]
    pages_sweep = [int(x) for x in args.pages_sweep.split(",") if x.strip()] if args.pages_sweep else []
    cache = PdfPageCache(args.dpi)

    throughput_results = []
    multipage_results = []
    ttft_results = []

    for profile_name, _, _ in PROFILES:
        rep = reps.get(profile_name)
        if not rep:
            print(f"\n--- [{name}] profile={profile_name}: 语料中未找到匹配代表文档，跳过 ---")
            continue

        tr = await bench_profile_throughput(
            name, base_url, model, profile_name, rep, cache, levels,
            args.requests_per_level, args.max_tokens, args.timeout,
            args.p99_threshold_ms, args.error_rate_threshold,
        )
        throughput_results.append(tr)

        if pages_sweep:
            mr = await bench_profile_multipage(
                name, base_url, model, profile_name, rep, cache, pages_sweep,
                args.max_pages_cap, args.max_tokens, args.timeout,
            )
            multipage_results.append(mr)

        if args.measure_ttft:
            tt = await bench_profile_ttft(
                name, base_url, model, profile_name, rep, cache,
                args.ttft_samples, args.max_tokens, args.timeout,
            )
            ttft_results.append(tt)

    return {
        "name": name, "base_url": base_url, "model": model,
        "throughput": throughput_results,
        "multipage_capacity": multipage_results,
        "ttft": ttft_results,
    }


async def main_async(args):
    print(f"扫描语料目录: {args.pdf_dir}")
    catalog = scan_catalog(args.pdf_dir)
    print(f"共 {len(catalog)} 个 PDF 文件，按大小分档:")
    for tier_name, lo, hi in SIZE_TIERS:
        cnt = sum(1 for _, s in catalog if lo <= s < hi)
        hi_str = "∞" if hi == float("inf") else f"{hi/1024/1024:.0f}MB"
        print(f"  {tier_name:<8s} [{lo/1024/1024:.2f}MB, {hi_str})  count={cnt}")

    print("\n挑选代表文档 (大小 x 篇幅/页数)...")
    reps = pick_representatives(catalog, max_attempts_per_profile=args.max_scan_attempts)
    for profile_name, size_tier, page_tier in PROFILES:
        rep = reps.get(profile_name)
        if rep:
            print(
                f"  {profile_name:<18s} <- {os.path.basename(rep['path'])[:50]}  "
                f"size={rep['size']/1024/1024:.2f}MB pages={rep['pages']} chapters={rep['chapters']}"
            )
        else:
            print(f"  {profile_name:<18s} <- 未找到匹配文档 (size={size_tier}, pages={page_tier})")

    targets = parse_targets(args.target) if args.target else DEFAULT_TARGETS
    summary = []
    for name, url, model in targets:
        res = await bench_target(name, url, model, reps, args)
        summary.append(res)

    print("\n=== 总结 ===")
    for s in summary:
        print(f"  [{s['name']}]")
        for tr in s["throughput"]:
            print(f"    {tr['profile']} (size={tr['size_bytes']/1024/1024:.1f}MB, pages={tr['pages']}): "
                  f"建议最大支撑并发 = {tr['recommended_concurrency']}")
        for mr in s["multipage_capacity"]:
            print(f"    {mr['profile']}: 单请求最大支撑页数 ≈ {mr['max_supported_pages']} "
                  f"(语料共{mr['pages_available']}页)")
        for tt in s["ttft"]:
            if tt["ttft_avg_ms"] is not None:
                print(f"    {tt['profile']}: TTFT avg = {tt['ttft_avg_ms']:.0f}ms")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"catalog_summary": {t[0]: sum(1 for _, s in catalog if t[1] <= s < t[2])
                                            for t in SIZE_TIERS},
                       "representatives": reps, "targets": summary}, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已写入: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="用真实 PDF 语料压测文档解析 VLM vLLM 服务")
    parser.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR, help="PDF 语料目录")
    parser.add_argument("--target", action="append",
                         help="格式: name=http://host:port,model_name，可重复指定，不指定则用默认两个服务")
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
                         help="单请求多页(真实连续页)容量探测的页数梯度；置空字符串跳过该测试")
    parser.add_argument("--max-pages-cap", type=int, default=32,
                         help="多页容量探测的绝对页数上限(避免超长文档渲染耗时过久)，默认 32")
    parser.add_argument("--dpi", type=int, default=150, help="PDF 页面渲染 DPI，默认 150 (常见扫描件分辨率)")
    parser.add_argument("--measure-ttft", action="store_true", default=True,
                         help="是否测量首 token 延迟，默认开启")
    parser.add_argument("--no-ttft", dest="measure_ttft", action="store_false", help="关闭 TTFT 测量")
    parser.add_argument("--ttft-samples", type=int, default=5, help="TTFT 采样请求数，默认 5")
    parser.add_argument("--max-scan-attempts", type=int, default=60,
                         help="为每个 profile 挑代表文档时最多尝试打开的候选文件数，默认 60")
    parser.add_argument("--output", default=None, help="将详细结果写入 JSON 文件路径")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
