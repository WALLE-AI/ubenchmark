#!/usr/bin/env bash
set -Eeuo pipefail

# 对比 Qwen3.8-27B 两种部署方案的在线服务性能：
#   A) PD分离（4卡：prefill 2,3 + decode 4,5），入口 http://127.0.0.1:19121
#   B) 单实例（2卡：0,1，TP=2），入口         http://127.0.0.1:19122
# 两者共用同一 sglang_env 环境的 bench_serving 工具，同一数据分布，
# 按并发梯度扫描，结果落盘到 bench_results/ 便于逐项对比。

PYTHON_BIN="${PYTHON_BIN:-/home/dataset1/gaojing/models/deploy/sglang_env/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-Qwen3.8-27B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/dataset0/images/Qwen3.8-27B}"
PD_PORT="${PD_PORT:-19121}"
GPU2_PORT="${GPU2_PORT:-19122}"

RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1024}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-256}"
NUM_PROMPTS_PER_LEVEL="${NUM_PROMPTS_PER_LEVEL:-64}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 4 8 16}"

OUT_DIR="${OUT_DIR:-./bench_results}"
mkdir -p "$OUT_DIR"

# random数据集会联网下载语料文件，本机代理访问huggingface会504超时；
# random-ids纯本地生成token id，且需强制离线模式避免tokenizer元数据解析卡死。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log() { echo "[$(date '+%F %T')] $*"; }

run_bench() {
    local tag=$1 port=$2 concurrency=$3
    local out="$OUT_DIR/${tag}_c${concurrency}.json"
    log ">>> [$tag] concurrency=$concurrency port=$port"
    "$PYTHON_BIN" -m sglang.bench_serving \
        --backend sglang \
        --host 127.0.0.1 --port "$port" \
        --model "$MODEL_NAME" \
        --tokenizer "$TOKENIZER_PATH" \
        --dataset-name random-ids \
        --random-input-len "$RANDOM_INPUT_LEN" \
        --random-output-len "$RANDOM_OUTPUT_LEN" \
        --random-range-ratio 1.0 \
        --num-prompts "$NUM_PROMPTS_PER_LEVEL" \
        --max-concurrency "$concurrency" \
        --seed 42 \
        --output-file "$out" \
        --disable-tqdm
}

for c in $CONCURRENCY_LEVELS; do
    run_bench pd "$PD_PORT" "$c"
    run_bench gpu2 "$GPU2_PORT" "$c"
done

log ">>> 汇总对比"
"$PYTHON_BIN" - "$OUT_DIR" "$CONCURRENCY_LEVELS" <<'PYEOF'
import json, sys, glob, os

out_dir = sys.argv[1]
levels = sys.argv[2].split()

FIELDS = [
    ("request_throughput", "req/s"),
    ("input_throughput", "tok/s(in)"),
    ("output_throughput", "tok/s(out)"),
    ("mean_ttft_ms", "TTFT均值ms"),
    ("median_ttft_ms", "TTFT中位ms"),
    ("mean_tpot_ms", "TPOT均值ms"),
    ("p99_ttft_ms", "TTFT p99 ms"),
]

def load(tag, c):
    path = os.path.join(out_dir, f"{tag}_c{c}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        # bench_serving 以追加JSON行写文件，取最后一行
        lines = [l for l in f if l.strip()]
        return json.loads(lines[-1]) if lines else None

header = ["concurrency", "方案"] + [name for _, name in FIELDS]
print(" | ".join(header))
print(" | ".join(["---"] * len(header)))
for c in levels:
    for tag, label in [("pd", "PD分离(4卡)"), ("gpu2", "单实例(2卡)")]:
        d = load(tag, c)
        if d is None:
            print(f"{c} | {label} | (缺少结果，检查 {out_dir}/{tag}_c{c}.json)")
            continue
        row = [str(c), label] + [f"{d.get(k, 'NA'):.2f}" if isinstance(d.get(k), (int, float)) else "NA" for k, _ in FIELDS]
        print(" | ".join(row))
PYEOF

log ">>> 完成。原始结果见 $OUT_DIR/*.json"
