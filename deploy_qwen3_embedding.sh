#!/usr/bin/env bash
# 在 GPU0/GPU1 上分别部署 Qwen3-Embedding-0.6B / Qwen3-Embedding-4B (vLLM OpenAI 兼容服务)
# 用法: ./deploy_qwen3_embedding.sh {start|stop|restart|status|logs} [0.6b|4b]

set -uo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/home/dataset-s3-0/gaojing/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-vllmEnvLatest}"

LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
PID_DIR="${PID_DIR:-${BASE_DIR}/pids}"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# ---- 0.6B ----
M06_NAME="qwen3-embedding-0.6b"
M06_PATH="${BASE_DIR}/Qwen3-Embedding-0.6B"
M06_GPU="${M06_GPU:-0}"
M06_PORT="${M06_PORT:-8001}"
M06_GPU_UTIL="${M06_GPU_UTIL:-0.25}"      # 40G 卡上约 10GB 显存上限
M06_MAX_LEN="${M06_MAX_LEN:-4096}"

# ---- 4B ----
M4B_NAME="qwen3-embedding-4b"
M4B_PATH="${BASE_DIR}/Qwen3-Embedding-4B"
M4B_GPU="${M4B_GPU:-1}"
M4B_PORT="${M4B_PORT:-8002}"
M4B_GPU_UTIL="${M4B_GPU_UTIL:-0.5}"       # 40G 卡上约 20GB 显存上限
M4B_MAX_LEN="${M4B_MAX_LEN:-4096}"

DTYPE="${DTYPE:-bfloat16}"

_pid_file() { echo "${PID_DIR}/$1.pid"; }
_log_file() { echo "${LOG_DIR}/$1.log"; }

_is_running() {
  local pid_file; pid_file="$(_pid_file "$1")"
  [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null
}

_start_one() {
  local name="$1" model_path="$2" gpu="$3" port="$4" gpu_util="$5" max_len="$6"

  if _is_running "${name}"; then
    echo "[${name}] 已在运行 (PID $(cat "$(_pid_file "${name}")"))，跳过。"
    return 0
  fi

  if [[ ! -d "${model_path}" ]]; then
    echo "[${name}] 模型目录不存在: ${model_path}" >&2
    return 1
  fi

  echo "[${name}] 启动中 -> GPU ${gpu}, port ${port}, gpu-memory-utilization ${gpu_util}"
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"

  CUDA_VISIBLE_DEVICES="${gpu}" nohup vllm serve "${model_path}" \
    --served-model-name "${name}" \
    --port "${port}" \
    --runner pooling \
    --convert embed \
    --dtype "${DTYPE}" \
    --gpu-memory-utilization "${gpu_util}" \
    --max-model-len "${max_len}" \
    --trust-remote-code \
    > "$(_log_file "${name}")" 2>&1 &

  echo $! > "$(_pid_file "${name}")"
  conda deactivate

  echo "[${name}] PID $(cat "$(_pid_file "${name}")")，日志: $(_log_file "${name}")"
}

_wait_ready() {
  # 4B 模型首次启动含 torch.compile，耗时可达 3~4 分钟，之后会命中编译缓存明显加快
  local name="$1" port="$2" timeout="${3:-360}"
  local waited=0
  echo "[${name}] 等待服务就绪 (最长 ${timeout}s)..."
  while (( waited < timeout )); do
    if ! _is_running "${name}"; then
      echo "[${name}] 进程已退出，启动失败，请查看日志: $(_log_file "${name}")" >&2
      return 1
    fi
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[${name}] 已就绪: http://127.0.0.1:${port}"
      return 0
    fi
    sleep 3; waited=$((waited + 3))
  done
  echo "[${name}] 等待超时，请查看日志: $(_log_file "${name}")" >&2
  return 1
}

_stop_one() {
  local name="$1"
  local pid_file; pid_file="$(_pid_file "${name}")"
  if ! _is_running "${name}"; then
    echo "[${name}] 未在运行。"
    rm -f "${pid_file}"
    return 0
  fi
  local pid; pid="$(cat "${pid_file}")"
  echo "[${name}] 停止 PID ${pid} ..."
  kill "${pid}" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 1
  done
  kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null
  rm -f "${pid_file}"
  echo "[${name}] 已停止。"
}

_status_one() {
  local name="$1" port="$2" gpu="$3"
  if _is_running "${name}"; then
    local pid; pid="$(cat "$(_pid_file "${name}")")"
    local mem; mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "${gpu}" 2>/dev/null)"
    echo "[${name}] RUNNING  PID=${pid}  GPU${gpu} 显存占用=${mem:-N/A}  http://127.0.0.1:${port}"
  else
    echo "[${name}] STOPPED"
  fi
}

TARGET="${2:-all}"

case "${1:-}" in
  start)
    rc=0
    if [[ "${TARGET}" == "all" || "${TARGET}" == "0.6b" ]]; then
      _start_one "${M06_NAME}" "${M06_PATH}" "${M06_GPU}" "${M06_PORT}" "${M06_GPU_UTIL}" "${M06_MAX_LEN}" || rc=1
    fi
    if [[ "${TARGET}" == "all" || "${TARGET}" == "4b" ]]; then
      _start_one "${M4B_NAME}" "${M4B_PATH}" "${M4B_GPU}" "${M4B_PORT}" "${M4B_GPU_UTIL}" "${M4B_MAX_LEN}" || rc=1
    fi
    if [[ "${TARGET}" == "all" || "${TARGET}" == "0.6b" ]]; then
      _wait_ready "${M06_NAME}" "${M06_PORT}" || rc=1
    fi
    if [[ "${TARGET}" == "all" || "${TARGET}" == "4b" ]]; then
      _wait_ready "${M4B_NAME}" "${M4B_PORT}" || rc=1
    fi
    exit "${rc}"
    ;;
  stop)
    [[ "${TARGET}" == "all" || "${TARGET}" == "0.6b" ]] && _stop_one "${M06_NAME}"
    [[ "${TARGET}" == "all" || "${TARGET}" == "4b" ]]   && _stop_one "${M4B_NAME}"
    exit 0
    ;;
  restart)
    "$0" stop "${TARGET}"
    "$0" start "${TARGET}"
    ;;
  status)
    _status_one "${M06_NAME}" "${M06_PORT}" "${M06_GPU}"
    _status_one "${M4B_NAME}" "${M4B_PORT}" "${M4B_GPU}"
    ;;
  logs)
    if [[ "${TARGET}" == "4b" ]]; then
      tail -f "$(_log_file "${M4B_NAME}")"
    else
      tail -f "$(_log_file "${M06_NAME}")"
    fi
    ;;
  *)
    echo "用法: $0 {start|stop|restart|status|logs} [0.6b|4b|all]"
    echo "  环境变量可覆盖: M06_GPU/M06_PORT/M06_GPU_UTIL/M06_MAX_LEN, M4B_GPU/M4B_PORT/M4B_GPU_UTIL/M4B_MAX_LEN, CONDA_ENV"
    exit 1
    ;;
esac
