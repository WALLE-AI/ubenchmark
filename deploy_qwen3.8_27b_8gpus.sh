#!/usr/bin/env bash
set -Eeuo pipefail

# Qwen3.8-27B 单实例部署（非PD分离），用于与 deploy_qwen3.8_pd.sh 的
# 4卡PD分离方案（端口19121）做性能对比。默认落在 GPU 0,1（TP=2），
# 与PD方案的GPU 2,3,4,5互不冲突，可同时启动两套服务对比压测。

PYTHON_BIN="${PYTHON_BIN:-/home/dataset1/gaojing/models/deploy/sglang_env/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/dataset0/images/Qwen3.8-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-19122}"                       # PD方案占用19121，此处默认换端口以便共存对比
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
TP_SIZE="${TP_SIZE:-2}"

# 与PD脚本保持一致的容量/调度参数，保证对比时输入条件可比：
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-$MAX_MODEL_LEN}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.6}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-512}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-lpm}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
LIMIT_MM_DATA="${LIMIT_MM_DATA:-{\"image\": 10, \"video\": 2\}}"
LOG_LEVEL="${LOG_LEVEL:-info}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

[ -x "${PYTHON_BIN}" ] || { echo "错误：找不到python解释器：${PYTHON_BIN}" >&2; exit 1; }

exec "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp-size "${TP_SIZE}" \
    --context-length "${MAX_MODEL_LEN}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --mamba-full-memory-ratio "${MAMBA_FULL_MEMORY_RATIO}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --max-queued-requests "${MAX_QUEUED_REQUESTS}" \
    --schedule-policy "${SCHEDULE_POLICY}" \
    --reasoning-parser "${REASONING_PARSER}" \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --limit-mm-data-per-request "${LIMIT_MM_DATA}" \
    --enable-cache-report \
    --enable-metrics \
    --collect-tokens-histogram \
    --log-level "${LOG_LEVEL}" \
    --log-requests \
    --log-requests-level 0 \
    --decode-log-interval 100
