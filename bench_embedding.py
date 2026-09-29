#!/usr/bin/env python3
"""
压测 vLLM OpenAI 兼容 embedding 服务 (/v1/embeddings)。
对每个服务，按并发梯度 (--concurrency-levels) 施压，统计:
  - 吞吐 (req/s, samples/s)
  - 延迟 P50/P90/P99/avg/max
  - 错误率
并给出建议的"最大支撑并发" (在延迟/错误率开始明显劣化前的最后一个健康档位)。

用法:
  python3 bench_embedding.py \
    --target 0.6b=http://127.0.0.1:8001,qwen3-embedding-0.6b \
    --target 4b=http://127.0.0.1:8002,qwen3-embedding-4b \
    --concurrency-levels 1,2,4,8,16,32,64 \
    --requests-per-level 200 \
    --batch-size 1 \
    --input-len short

也可用默认配置直接跑两个已部署的模型:
  python3 bench_embedding.py
"""

import argparse
import asyncio
import json
import statistics
import time

import aiohttp

DEFAULT_TARGETS = [
    ("0.6b", "http://127.0.0.1:8001", "qwen3-embedding-0.6b"),
    ("4b", "http://127.0.0.1:8002", "qwen3-embedding-4b"),
]

SAMPLE_TEXTS = {
    "short": "人工智能正在快速改变软件开发的方式。",
    "medium": "人工智能正在快速改变软件开发的方式。" * 20,
    "long": "人工智能正在快速改变软件开发的方式。" * 100,
}


def parse_targets(raw_targets):
    targets = []
    for item in raw_targets:
        name, rest = item.split("=", 1)
        url, model = rest.split(",", 1)
        targets.append((name.strip(), url.strip().rstrip("/"), model.strip()))
    return targets


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


async def send_one(session, url, payload, timeout_s):
    t0 = time.perf_counter()
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as resp:
            body = await resp.read()
            dt = time.perf_counter() - t0
            if resp.status != 200:
                return dt, False, f"HTTP {resp.status}: {body[:200]!r}"
            data = json.loads(body)
            if "data" not in data:
                return dt, False, f"malformed response: {body[:200]!r}"
            return dt, True, None
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t0
        return dt, False, f"{type(e).__name__}: {e}"


async def run_level(base_url, model, concurrency, total_requests, batch_size, text, timeout_s):
    url = f"{base_url}/v1/embeddings"
    payload_input = [text] * batch_size if batch_size > 1 else text
    payload = {"model": model, "input": payload_input}

    sem = asyncio.Semaphore(concurrency)
    latencies = []
    errors = []

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:

        async def worker():
            async with sem:
                dt, ok, err = await send_one(session, url, payload, timeout_s)
                latencies.append(dt)
                if not ok:
                    errors.append(err)

        wall_t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(total_requests)))
        wall_dt = time.perf_counter() - wall_t0

    n_ok = total_requests - len(errors)
    req_per_s = n_ok / wall_dt if wall_dt > 0 else 0.0
    samples_per_s = req_per_s * batch_size

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "ok": n_ok,
        "errors": len(errors),
        "error_samples": errors[:3],
        "wall_time_s": wall_dt,
        "req_per_s": req_per_s,
        "samples_per_s": samples_per_s,
        "latency_avg_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
        "latency_p50_ms": percentile(latencies, 50) * 1000,
        "latency_p90_ms": percentile(latencies, 90) * 1000,
        "latency_p99_ms": percentile(latencies, 99) * 1000,
        "latency_max_ms": max(latencies) * 1000 if latencies else 0.0,
    }


async def warmup(base_url, model, text, timeout_s):
    url = f"{base_url}/v1/embeddings"
    async with aiohttp.ClientSession() as session:
        await send_one(session, url, {"model": model, "input": text}, timeout_s)


def print_level_result(r):
    err_rate = (r["errors"] / r["total_requests"] * 100) if r["total_requests"] else 0
    print(
        f"  C={r['concurrency']:<4d} "
        f"req/s={r['req_per_s']:7.2f}  samples/s={r['samples_per_s']:7.2f}  "
        f"p50={r['latency_p50_ms']:7.1f}ms p90={r['latency_p90_ms']:7.1f}ms "
        f"p99={r['latency_p99_ms']:7.1f}ms max={r['latency_max_ms']:8.1f}ms  "
        f"err={r['errors']}/{r['total_requests']} ({err_rate:.1f}%)"
    )
    if r["error_samples"]:
        for e in r["error_samples"]:
            print(f"      ! {e}")


def pick_max_supported(results, p99_threshold_ms, error_rate_threshold):
    """选最后一个 '健康' 档位: p99 未超阈值 且 错误率未超阈值，且吞吐相对更高并发未显著回退。"""
    healthy = []
    for r in results:
        err_rate = (r["errors"] / r["total_requests"]) if r["total_requests"] else 1.0
        if err_rate <= error_rate_threshold and r["latency_p99_ms"] <= p99_threshold_ms:
            healthy.append(r)
    if not healthy:
        return None
    return max(healthy, key=lambda r: r["concurrency"])


async def bench_target(name, base_url, model, levels, requests_per_level, batch_size,
                        input_len, timeout_s, p99_threshold_ms, error_rate_threshold):
    text = SAMPLE_TEXTS[input_len]
    print(f"\n=== [{name}] {base_url}  model={model}  input_len={input_len}({len(text)} chars)  batch_size={batch_size} ===")

    print("  预热中...")
    try:
        await warmup(base_url, model, text, timeout_s)
    except Exception as e:  # noqa: BLE001
        print(f"  预热失败: {e}")

    results = []
    for c in levels:
        n = max(requests_per_level, c)  # 保证每个并发档至少打满一轮
        r = await run_level(base_url, model, c, n, batch_size, text, timeout_s)
        print_level_result(r)
        results.append(r)

    best = pick_max_supported(results, p99_threshold_ms, error_rate_threshold)
    print(f"  --- [{name}] 汇总 ---")
    if best:
        print(
            f"  建议最大支撑并发 ≈ {best['concurrency']} "
            f"(p99={best['latency_p99_ms']:.1f}ms, req/s={best['req_per_s']:.2f}, "
            f"错误率={(best['errors']/best['total_requests']*100):.1f}%)"
        )
    else:
        print("  未找到满足阈值的健康并发档位，请降低并发或检查服务日志。")

    peak = max(results, key=lambda r: r["samples_per_s"])
    print(
        f"  峰值吞吐 ≈ {peak['samples_per_s']:.2f} samples/s "
        f"(在并发={peak['concurrency']} 时)"
    )
    return {"name": name, "results": results, "recommended_concurrency": best["concurrency"] if best else None}


async def main_async(args):
    if args.target:
        targets = parse_targets(args.target)
    else:
        targets = DEFAULT_TARGETS

    levels = [int(x) for x in args.concurrency_levels.split(",") if x.strip()]

    summary = []
    for name, url, model in targets:
        res = await bench_target(
            name, url, model, levels, args.requests_per_level, args.batch_size,
            args.input_len, args.timeout, args.p99_threshold_ms, args.error_rate_threshold,
        )
        summary.append(res)

    print("\n=== 总结 ===")
    for s in summary:
        print(f"  {s['name']}: 建议最大支撑并发 = {s['recommended_concurrency']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已写入: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="压测 Qwen3 Embedding vLLM 服务")
    parser.add_argument(
        "--target", action="append",
        help="格式: name=http://host:port,model_name，可重复指定多个，不指定则用默认两个服务",
    )
    parser.add_argument("--concurrency-levels", default="1,2,4,8,16,32,64,128",
                         help="逗号分隔的并发梯度，默认 1,2,4,8,16,32,64,128")
    parser.add_argument("--requests-per-level", type=int, default=100,
                         help="每个并发档发送的请求总数（不足并发数时取并发数），默认 100")
    parser.add_argument("--batch-size", type=int, default=1,
                         help="单次请求内 input 的样本数（batch embedding），默认 1")
    parser.add_argument("--input-len", choices=list(SAMPLE_TEXTS.keys()), default="short",
                         help="文本长度: short/medium/long，默认 short")
    parser.add_argument("--timeout", type=float, default=60.0, help="单请求超时(s)，默认 60")
    parser.add_argument("--p99-threshold-ms", type=float, default=2000.0,
                         help="判定'健康'档位的 p99 延迟阈值(ms)，默认 2000")
    parser.add_argument("--error-rate-threshold", type=float, default=0.01,
                         help="判定'健康'档位的最大错误率，默认 0.01 (1%)")
    parser.add_argument("--output", default=None, help="将详细结果写入 JSON 文件路径")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
