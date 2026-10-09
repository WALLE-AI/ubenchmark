#!/usr/bin/env bash
# 在 GPU0/GPU1 上分别部署 MinerU2.5-Pro-2605-1.2B / TeleOCR (vLLM OpenAI 兼容服务)
# 两者均为 Qwen2(.5)-VL 结构的文档解析 VLM，走 chat/completions 接口，支持图片输入。
# 用法: ./deploy_ocr_vlm.sh {start|stop|restart|status|logs} [mineru|teleocr|all]

set -uo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="/home/dataset-s3-0/gaojing/models"
CONDA_SH="/home/dataset-s3-0/gaojing/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-vllmEnvLatest}"

LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
PID_DIR="${PID_DIR:-${BASE_DIR}/pids}"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# Triton JIT 编译缓存目录：默认 ~/.triton/cache 属主为 uid 1002（775），
# 当前用户无写权限会导致 EngineCore 启动时 PermissionError，这里改到项目内可写目录。
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${BASE_DIR}/.triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"

# ---- MinerU2.5-Pro-2605-1.2B ----
MU_NAME="mineru2.5-pro-1.2b"
MU_PATH="${MODELS_DIR}/MinerU2.5-Pro-2605-1.2B"
MU_GPU="${MU_GPU:-0}"
MU_PORT="${MU_PORT:-8003}"
MU_GPU_UTIL="${MU_GPU_UTIL:-0.3}"        # GPU0 已有占用，预留约 12GB 显存上限
MU_MAX_LEN="${MU_MAX_LEN:-8192}"

# ---- TeleOCR ----
TO_NAME="teleocr"
TO_PATH="${MODELS_DIR}/TeleOCR"
TO_GPU="${TO_GPU:-1}"
TO_PORT="${TO_PORT:-8004}"
TO_GPU_UTIL="${TO_GPU_UTIL:-0.25}"       # GPU1 已有较多占用，预留约 10GB 显存上限
TO_MAX_LEN="${TO_MAX_LEN:-8192}"

DTYPE="${DTYPE:-bfloat16}"

# TeleOCR 语言模型为 Qwen3 结构 (head_dim=128 + qk-norm)，与 qwen2_5_vl 默认的
# Qwen2ForCausalLM 不符，需加载官方 out-of-tree 插件 TeleOCR-vllm 覆盖该架构注册。
# 插件无法写入 conda env (site-packages 属主为 uid 1002)，故用 pip --target 装到
# 项目目录并经 PYTHONPATH 注入。
TO_PLUGIN_DIR="${TO_PLUGIN_DIR:-${BASE_DIR}/.vllm_plugins}"

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

  # TeleOCR 需要额外加载 TeleOCR-vllm 插件 (见上方说明)
  local extra_env=()
  if [[ "${name}" == "${TO_NAME}" ]]; then
    if [[ ! -d "${TO_PLUGIN_DIR}/TeleOCR_vllm" ]]; then
      echo "[${name}] 未找到插件目录: ${TO_PLUGIN_DIR}" >&2
      return 1
    fi
    extra_env=(PYTHONPATH="${TO_PLUGIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}")
    echo "[${name}] 启用 TeleOCR-vllm 插件: ${TO_PLUGIN_DIR}"
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" env "${extra_env[@]}" nohup vllm serve "${model_path}" \
    --served-model-name "${name}" \
    --port "${port}" \
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
    if [[ "${TARGET}" == "all" || "${TARGET}" == "mineru" ]]; then
      _start_one "${MU_NAME}" "${MU_PATH}" "${MU_GPU}" "${MU_PORT}" "${MU_GPU_UTIL}" "${MU_MAX_LEN}" || rc=1
    fi
    if [[ "${TARGET}" == "all" || "${TARGET}" == "teleocr" ]]; then
      _start_one "${TO_NAME}" "${TO_PATH}" "${TO_GPU}" "${TO_PORT}" "${TO_GPU_UTIL}" "${TO_MAX_LEN}" || rc=1
    fi
    if [[ "${TARGET}" == "all" || "${TARGET}" == "mineru" ]]; then
      _wait_ready "${MU_NAME}" "${MU_PORT}" || rc=1
    fi
    if [[ "${TARGET}" == "all" || "${TARGET}" == "teleocr" ]]; then
      _wait_ready "${TO_NAME}" "${TO_PORT}" || rc=1
    fi
    exit "${rc}"
    ;;
  stop)
    [[ "${TARGET}" == "all" || "${TARGET}" == "mineru" ]]  && _stop_one "${MU_NAME}"
    [[ "${TARGET}" == "all" || "${TARGET}" == "teleocr" ]] && _stop_one "${TO_NAME}"
    exit 0
    ;;
  restart)
    "$0" stop "${TARGET}"
    "$0" start "${TARGET}"
    ;;
  status)
    _status_one "${MU_NAME}" "${MU_PORT}" "${MU_GPU}"
    _status_one "${TO_NAME}" "${TO_PORT}" "${TO_GPU}"
    ;;
  logs)
    if [[ "${TARGET}" == "teleocr" ]]; then
      tail -f "$(_log_file "${TO_NAME}")"
    else
      tail -f "$(_log_file "${MU_NAME}")"
    fi
    ;;
  *)
    echo "用法: $0 {start|stop|restart|status|logs} [mineru|teleocr|all]"
    echo "  环境变量可覆盖: MU_GPU/MU_PORT/MU_GPU_UTIL/MU_MAX_LEN, TO_GPU/TO_PORT/TO_GPU_UTIL/TO_MAX_LEN, CONDA_ENV"
    exit 1
    ;;
esac
