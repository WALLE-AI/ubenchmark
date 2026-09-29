#!/bin/bash
# ============================================================
# Qwen3.8-27B PD分离部署脚本（生产环境）
#
# 硬件：单机8×A100-SXM4-40G | 拓扑：2卡Prefill + 2卡Decode（GPU 2,3 / 4,5）
# 框架：SGLang 0.5.10 | KV传输：单节点NVLink（INTRA_NODE_NVLINK，驱动580.178.04）
# 权重：bfloat16（注意：模型目录不含quantization_config，并非FP8；A100也不支持FP8算力）
#
# ------------------------------------------------------------
# 容量模型（来自实测日志，用于推导下面所有参数，改参数前请先读这段）
# ------------------------------------------------------------
# 模型结构：64层中仅16层full_attention，其余48层linear_attention(mamba)。
#   → KV Cache：32 KB/token（每卡分片，TP=2）
#   → Mamba State：约 75.7 MB/请求（ssm_state为float32）——这是并发的真正瓶颈
#
# 单卡显存账本（40960 MiB）：
#   权重 25.64 GB + CUDA/NCCL上下文 ~1.7 GB + 激活/传输缓冲 ~2.4 GB（在static预算之外）
#   → mem-fraction-static=0.80 实测仅用到 34.4 GB，白白空出 6.86 GB
#   → 本脚本提升到 0.86(prefill) / 0.90(decode)，显存池 5.1 GB → 约 8.3 / 9.1 GB
#     prefill激活峰值更大（视觉塔 + 不使用CUDA Graph），故留更多余量
#
# 显存池按 --mamba-full-memory-ratio 在 Mamba 与 KV 之间切分。
# 本脚本用 0.6（偏向KV），因为 radix 前缀缓存（prompt cache）住在KV池里，
# KV池越大 → 命中率越高、可支撑的上下文越长：
#   decode池 9.1 GB → Mamba 3.6 GB(48槽) + KV 5.5 GB(约 181k tokens)
#
# 并发上限的两条硬约束（SGLang源码 model_runner_kv_cache_mixin.py）：
#   - 开启radix cache时：max_running_requests = mamba槽位 / 3   → prefill端
#   - 关闭radix cache时：max_running_requests = mamba槽位       → decode端(PD强制关闭)
#   所以 prefill 16 并发 / decode 48 并发是匹配的：prefill是chunked计算密集型，
#   16路足以喂满2张A100；decode是显存带宽型，需要大batch才有吞吐。
#
# 上下文长度权衡（decode侧KV池约181k tokens）：
#   context-length  单请求满长占KV池   并发满长请求数   建议
#      32768            1.05 GB           ~19        高并发短对话
#      65536            2.10 GB            ~8        ★默认，通用
#     131072            4.19 GB            ~4        长文档，并发会明显下降
#     262144            8.39 GB            ~2        模型上限，不建议在40G卡上用
#
# ------------------------------------------------------------
# 用法
# ------------------------------------------------------------
#   ./deploy_qwen3.8_pd.sh                      # 前台启动 + 守护
#   MAX_MODEL_LEN=131072 ./deploy_qwen3.8_pd.sh # 覆盖任意参数
#   FORCE_RESTART=1 ./deploy_qwen3.8_pd.sh      # 端口被占时强制接管
#   SGLANG_API_KEY=xxx ./deploy_qwen3.8_pd.sh   # 开启鉴权
#   nohup ./deploy_qwen3.8_pd.sh > pd_deploy_run.log 2>&1 &
# ============================================================
set -euo pipefail

# ============================================================
# 1. 可配置参数（全部支持环境变量覆盖）
# ============================================================

# ---- 模型与运行时 ----
MODEL_PATH="${MODEL_PATH:-/home/dataset0/images/Qwen3.8-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"   # 对外模型名，客户端用这个而非绝对路径
PYTHON_BIN="${PYTHON_BIN:-/home/dataset1/gaojing/models/deploy/sglang_env/.venv/bin/python}"

# ---- GPU拓扑 ----
PREFILL_GPUS="${PREFILL_GPUS:-2,3}"
DECODE_GPUS="${DECODE_GPUS:-4,5}"
PREFILL_TP="${PREFILL_TP:-2}"
DECODE_TP="${DECODE_TP:-2}"

# ---- 端口 ----
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30001}"
ROUTER_PORT="${ROUTER_PORT:-19121}"        # 对外服务入口
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"   # PD握手端口（prefill侧监听）
PROMETHEUS_PORT="${PROMETHEUS_PORT:-29000}" # Router指标暴露端口

# ---- 上下文 / max_tokens ----
# context-length = 输入+输出总长上限，客户端 max_tokens 的天花板。
# 模型支持 262144，但受KV池限制（见上表），生产默认 65536。
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-$MAX_MODEL_LEN}"  # 必须 >= context-length，否则长prompt直接报错
# ★ 这个参数同时决定 prompt cache 的复用粒度（实测结论，见下方 Prompt Cache 段）：
#   前缀缓存只能按 chunked_prefill_size 的整数倍复用，prompt 短于该值则完全不命中。
#   实测(8192 tok 唯一prompt、输出64)：
#     8192 → 冷TTFT 1226ms(c=1)/16322ms(c=16)，但只有 ≥8192 tok 的prompt才吃到缓存
#     2048 → 冷TTFT 1513ms(c=1)/17397ms(c=16)，≥2048 tok 即可吃到缓存
#   冷启动慢约19%，但命中时快 4.3 倍（3002 tok: 1.258s → 0.294s），
#   对 RAG / 长system prompt 这类真实负载净收益明显，故默认 2048。
#   若你的负载全是互不相同的长文档（无共享前缀），改回 8192 更快。
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
# 1=超长prompt自动截断（保可用性），0=返回400（保正确性）。生产默认显式报错。
ALLOW_AUTO_TRUNCATE="${ALLOW_AUTO_TRUNCATE:-0}"

# ---- 显存 ----
PREFILL_MEM_FRACTION="${PREFILL_MEM_FRACTION:-0.86}"
DECODE_MEM_FRACTION="${DECODE_MEM_FRACTION:-0.90}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.6}"   # Mamba显存 / KV显存，越小则KV池越大
# float32(默认，跟随模型config) → bfloat16 可让Mamba状态减半、并发翻倍，但改变数值精度。
# 生产环境建议先离线评测再开启：MAMBA_SSM_DTYPE=bfloat16
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-}"

# ---- 并发 ----
# prefill: 受 mamba槽位/3 约束，16路足以喂满2×A100（chunked prefill是计算密集型）
# decode : 显式设为48 → 直接决定 mamba槽位=48（radix关闭时的代码路径），显存已核算 3.6 GB
PREFILL_MAX_RUNNING="${PREFILL_MAX_RUNNING:-16}"
DECODE_MAX_RUNNING="${DECODE_MAX_RUNNING:-48}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-512}"      # 引擎侧排队上限，超出立即拒绝而非无限堆积
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-48}"          # 必须 >= DECODE_MAX_RUNNING，否则大batch退化为eager
ROUTER_MAX_CONCURRENT="${ROUTER_MAX_CONCURRENT:-64}"  # 网关准入上限（略高于decode并发，容纳传输中的请求）
ROUTER_QUEUE_SIZE="${ROUTER_QUEUE_SIZE:-512}"
ROUTER_QUEUE_TIMEOUT="${ROUTER_QUEUE_TIMEOUT:-300}"
REQUEST_TIMEOUT_SECS="${REQUEST_TIMEOUT_SECS:-3600}"  # 长文本生成需要足够长

# ---- Prompt Cache（前缀缓存）----
# ✅ 实测结论（2026-09-29，SGLang 0.5.10 + Qwen3.8-27B）：prompt cache 确实生效，
#    但【复用粒度 = chunked_prefill_size】，这是本模型最反直觉的一点：
#      prompt 长度 < chunked_prefill_size  → 完全不命中（cached=0）
#      prompt 长度 ≥ chunked_prefill_size  → 按其整数倍复用
#    实测(chunked=2048)：3002 tok → 命中2048；4802 → 4096；16502 → 16384
#    实测(chunked=8192)：3002/4802/6002 tok 全部命中 0，9002 → 命中8192
#    收益：3002 tok 的重复前缀请求 1.258s → 0.294s（快 4.3 倍）
#
#    ⚠️ 排查陷阱（我踩过，别重复）：
#      1) 用"完全相同的整条prompt"连发两次 → cached 恒为 0（末尾必须留至少1个新token，
#         真实业务的问题各不相同，天然满足；自测时请让后缀不同）
#      2) Router 的 /v1/chat/completions 响应【不返回】prompt_tokens_details，
#         看到 None 不代表没命中。权威信号只有两个：
#           grep "Prefill batch" pd_logs/prefill.log   → 看 #cached-token
#           curl -s http://127.0.0.1:30000/metrics | grep cache_hit_rate
#      3) cache_hit_rate 是累计平均值，冷请求会把它拉低，别只看一次。
# 四层配置：
#   1) prefill引擎 radix cache（默认开启，勿加 --disable-radix-cache）
#   2) --schedule-policy lpm：按最长前缀匹配排序，让同前缀请求相邻调度
#   3) Router --prefill-policy cache_aware：多prefill实例时把同前缀请求路由到同一实例
#   4) --enable-cache-report：响应 usage.prompt_tokens_details.cached_tokens 回传命中数
SCHEDULE_POLICY="${SCHEDULE_POLICY:-lpm}"
# hybrid mamba 调度策略（仅prefill侧；decode侧被PD强制关闭radix，设extra_buffer会直接报错）：
#   no_buffer   ：SGLang默认。prefill并发 = mamba槽位/3（实测12）
#   extra_buffer：保留mamba state checkpoint，并额外打开overlap调度（吞吐略好），
#                 但并发 = mamba槽位/5（实测7）。实测对命中率无改善，故默认不用。
PREFILL_MAMBA_STRATEGY="${PREFILL_MAMBA_STRATEGY:-no_buffer}"
ROUTER_PREFILL_POLICY="${ROUTER_PREFILL_POLICY:-cache_aware}"
# 单decode实例只能用 random/round_robin/cache_aware；
# power_of_two（最小负载，多实例最优）要求 >=2 个decode worker，否则router拒绝启动。
ROUTER_DECODE_POLICY="${ROUTER_DECODE_POLICY:-round_robin}"
ROUTER_CACHE_THRESHOLD="${ROUTER_CACHE_THRESHOLD:-0.5}"
ROUTER_MAX_TREE_SIZE="${ROUTER_MAX_TREE_SIZE:-16777216}"
ROUTER_EVICTION_INTERVAL="${ROUTER_EVICTION_INTERVAL:-60}"

# ---- 解析器 / 多模态 ----
REASONING_PARSER="${REASONING_PARSER:-qwen3}"          # 分离 <think> 到 reasoning_content
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
LIMIT_MM_DATA="${LIMIT_MM_DATA:-{\"image\": 10, \"video\": 2\}}"  # 单请求多模态输入上限，防止OOM攻击

# ---- 可观测性 / 稳定性 ----
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-600}"            # 单次forward卡死多久后自杀（由守护逻辑重启）
LOG_LEVEL="${LOG_LEVEL:-info}"
LOG_REQUESTS_LEVEL="${LOG_REQUESTS_LEVEL:-0}"          # 0=仅元数据，不落盘prompt内容
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"              # 单实例启动等待上限（27B权重加载+图捕获）
SGLANG_API_KEY="${SGLANG_API_KEY:-}"                   # 非空则开启Bearer鉴权
FORCE_RESTART="${FORCE_RESTART:-0}"
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"

LOG_DIR="${LOG_DIR:-./pd_logs}"
RUN_DIR="${RUN_DIR:-./pd_run}"

# ============================================================
# 2. 环境变量
# ============================================================
# mooncake依赖libibverbs.so.1/libnl-route-3.so.200，本机无系统级rdma-core安装权限，
# 因此把对应.so放在venv旁边由LD_LIBRARY_PATH提供（不使用真实IB硬件，仅满足动态链接）。
export LD_LIBRARY_PATH="/home/dataset1/gaojing/models/deploy/sglang_env/extra-libs:${LD_LIBRARY_PATH:-}"

# 代理会拦截127.0.0.1的内部请求（prefill/decode/router互相探活），必须排除
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

# 单节点NVLink KV传输（驱动580.178.04支持cudaMemcpyBatchAsync）
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
export MC_INTRANODE_NVLINK=true

# PD握手超时：27B权重加载慢，握手窗口必须放宽
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1000000000}"
export SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL="${SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL:-10000000}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# 高并发下每个请求占用多个fd（HTTP + zmq + 传输），默认1024会在压测时耗尽
ulimit -n 65535 2>/dev/null || echo "警告：无法提升fd上限，高并发下可能出现 Too many open files"

mkdir -p "$LOG_DIR" "$RUN_DIR"

# ============================================================
# 3. 工具函数
# ============================================================
log()  { echo "[$(date '+%F %T')] $*"; }
fail() { echo "[$(date '+%F %T')] 错误：$*" >&2; exit 1; }

port_in_use() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3<&- ; return 0; } || return 1
}

# 只杀掉本脚本启动的进程组，绝不使用全局 pkill -f sglang
# （本机8卡上可能同时跑着其他模型部署，全局pkill会误杀）
kill_pidfile() {
    local f="$RUN_DIR/$1.pid" pid
    [ -f "$f" ] || return 0
    pid=$(cat "$f" 2>/dev/null || true)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        log "    停止 $1 (PGID $pid)"
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -KILL -- "-$pid" 2>/dev/null || true
    fi
    rm -f "$f"
}

# 按端口回收：用于接管"旧版脚本启动、没有PID文件"的实例。
# 仅杀掉正在监听我们自己端口的进程，不做全局 pkill，因此不会误伤其他模型部署。
kill_by_port() {
    local port=$1 pid pgid
    for pid in $(ss -ltnp 2>/dev/null \
                  | awk -v p=":$port" '$4 ~ p"$"' \
                  | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
        kill -0 "$pid" 2>/dev/null || continue
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
        # 自我保护：绝不杀掉本脚本自己所在的进程组
        if [ -n "${pgid:-}" ] && [ "$pgid" = "$(ps -o pgid= -p $$ | tr -d ' ')" ]; then
            log "    跳过端口 $port（PID=$pid 属于本脚本自身进程组）"
            continue
        fi
        log "    回收端口 $port 上的进程 PID=$pid PGID=${pgid:-?}"
        if [ -n "${pgid:-}" ]; then
            kill -TERM -- "-$pgid" 2>/dev/null || true
        else
            kill -TERM "$pid" 2>/dev/null || true
        fi
        for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -KILL "$pid" 2>/dev/null || true
    done
}

wait_ready() {  # wait_ready <名称> <端口> <路径>
    local name=$1 port=$2 path=$3 i
    log "    等待 $name 就绪（最多 ${STARTUP_TIMEOUT}s）..."
    for i in $(seq 1 "$STARTUP_TIMEOUT"); do
        if curl -sf --max-time 5 "http://127.0.0.1:${port}${path}" >/dev/null 2>&1; then
            log "    ✅ $name 就绪（耗时 ${i}s）"
            return 0
        fi
        # 进程已死就别再等了
        local pid; pid=$(cat "$RUN_DIR/${name}.pid" 2>/dev/null || true)
        if [ -n "${pid:-}" ] && ! kill -0 "$pid" 2>/dev/null; then
            fail "$name 进程已退出，请检查 $LOG_DIR/${name}.log"
        fi
        sleep 1
    done
    fail "$name 启动超时（${STARTUP_TIMEOUT}s），请检查 $LOG_DIR/${name}.log"
}

# ============================================================
# 4. 启动前检查
# ============================================================
log ">>> 启动前检查..."

[ -x "$PYTHON_BIN" ] || fail "找不到python解释器：$PYTHON_BIN"
"$PYTHON_BIN" -c "import sglang" 2>/dev/null \
    || fail "SGLang未安装于 $PYTHON_BIN，请执行：cd sglang_env && uv sync"
"$PYTHON_BIN" -c "import sglang_router" 2>/dev/null \
    || fail "sglang_router未安装于 $PYTHON_BIN"
SGLANG_VERSION=$("$PYTHON_BIN" -c "import sglang; print(sglang.__version__)" 2>/dev/null)

[ -d "$MODEL_PATH" ] || fail "模型路径不存在：$MODEL_PATH
请先下载： modelscope download --model Qwen/Qwen3.8-27B --local_dir $MODEL_PATH"
[ -f "$MODEL_PATH/config.json" ] || fail "模型目录缺少 config.json：$MODEL_PATH"

# GPU数量与显存检查：权重25.64GB/卡，起不来的话早点报错而不是等5分钟OOM
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
NEEDED_GPUS=$((PREFILL_TP + DECODE_TP))
[ "$GPU_COUNT" -ge "$NEEDED_GPUS" ] \
    || fail "检测到 $GPU_COUNT 张GPU，当前拓扑需要 $NEEDED_GPUS 张"

for gpu in ${PREFILL_GPUS//,/ } ${DECODE_GPUS//,/ }; do
    used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
    if [ "$used" -gt 2048 ]; then
        if [ "$FORCE_RESTART" = "1" ]; then
            log "    ⚠️  GPU $gpu 已占用 ${used} MiB（FORCE_RESTART=1，继续）"
        else
            fail "GPU $gpu 已被占用 ${used} MiB。先停掉旧服务，或用 FORCE_RESTART=1 强制接管"
        fi
    fi
done

# 端口检查
for p in "$PREFILL_PORT" "$DECODE_PORT" "$ROUTER_PORT" "$BOOTSTRAP_PORT" "$PROMETHEUS_PORT"; do
    if port_in_use "$p"; then
        [ "$FORCE_RESTART" = "1" ] \
            || fail "端口 $p 已被占用。先停掉旧服务，或用 FORCE_RESTART=1 强制接管"
        log "    ⚠️  端口 $p 已占用（FORCE_RESTART=1，尝试接管本脚本的旧进程）"
    fi
done

# 接管旧实例：先按PID文件，再按端口（兼容旧版脚本留下的无PID文件实例）
if [ "$FORCE_RESTART" = "1" ]; then
    log ">>> FORCE_RESTART=1：清理旧实例..."
    kill_pidfile router; kill_pidfile decode; kill_pidfile prefill
    kill_by_port "$ROUTER_PORT"; kill_by_port "$DECODE_PORT"; kill_by_port "$PREFILL_PORT"

    # 等显存真正释放，否则新实例会在加载权重时OOM
    log "    等待GPU显存释放..."
    for _ in $(seq 1 60); do
        busy=0
        for gpu in ${PREFILL_GPUS//,/ } ${DECODE_GPUS//,/ }; do
            used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
            [ "$used" -gt 2048 ] && busy=1
        done
        [ "$busy" = "0" ] && break
        sleep 2
    done
    [ "${busy:-0}" = "0" ] && log "    ✅ 显存已释放" \
                           || log "    ⚠️  显存仍被占用，继续启动（可能OOM）"

    # 静默期：被回收的旧实例可能是"带全局 pkill -f sglang 的旧版脚本"或外部supervisor，
    # 其退出钩子会在死亡后一两秒内误杀我们刚启动的新进程。等它彻底安静再启动。
    log "    等待旧supervisor退出钩子执行完毕（15s静默期）..."
    sleep 15
fi

log ">>> 检查通过"
log "    SGLang版本 : $SGLANG_VERSION"
log "    模型       : $MODEL_PATH  (对外名: $SERVED_MODEL_NAME)"
log "    Prefill    : GPU $PREFILL_GPUS  TP=$PREFILL_TP  mem=$PREFILL_MEM_FRACTION  并发=$PREFILL_MAX_RUNNING"
log "    Decode     : GPU $DECODE_GPUS  TP=$DECODE_TP  mem=$DECODE_MEM_FRACTION  并发=$DECODE_MAX_RUNNING"
log "    上下文     : $MAX_MODEL_LEN tokens（客户端 max_tokens 上限）"
log "    鉴权       : $([ -n "$SGLANG_API_KEY" ] && echo 已开启 || echo 未开启)"

# ============================================================
# 5. 退出清理
# ============================================================
SHUTTING_DOWN=0
cleanup() {
    [ "$SHUTTING_DOWN" = "1" ] && return
    SHUTTING_DOWN=1
    log ">>> 正在停止所有服务..."
    kill_pidfile router
    kill_pidfile decode
    kill_pidfile prefill
    log ">>> 所有服务已停止"
}
trap cleanup EXIT INT TERM

# ============================================================
# 6. 公共引擎参数
# ============================================================
COMMON_ARGS=(
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --host 127.0.0.1                      # 引擎只监听本地，只有Router对外暴露
    --context-length "$MAX_MODEL_LEN"
    --max-prefill-tokens "$MAX_PREFILL_TOKENS"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --max-queued-requests "$MAX_QUEUED_REQUESTS"
    --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO"
    --disaggregation-mode placeholder     # 由各角色覆盖
    --disaggregation-transfer-backend mooncake
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
    # OpenAI兼容层
    --reasoning-parser "$REASONING_PARSER"
    --tool-call-parser "$TOOL_CALL_PARSER"
    --limit-mm-data-per-request "$LIMIT_MM_DATA"
    --enable-cache-report                 # 回传 cached_tokens，用于监控prompt cache命中率
    # 可观测性与稳定性
    --enable-metrics                      # /metrics 暴露 Prometheus 指标
    --collect-tokens-histogram
    --watchdog-timeout "$WATCHDOG_TIMEOUT"
    --log-level "$LOG_LEVEL"
    --log-requests
    --log-requests-level "$LOG_REQUESTS_LEVEL"
    --decode-log-interval 100
)
[ -n "$SGLANG_API_KEY" ]        && COMMON_ARGS+=(--api-key "$SGLANG_API_KEY")
[ -n "$MAMBA_SSM_DTYPE" ]       && COMMON_ARGS+=(--mamba-ssm-dtype "$MAMBA_SSM_DTYPE")
[ "$ALLOW_AUTO_TRUNCATE" = "1" ] && COMMON_ARGS+=(--allow-auto-truncate)

# 把 --disaggregation-mode 的占位值替换成真实角色
role_args() {  # role_args <prefill|decode>
    local out=() a
    for a in "${COMMON_ARGS[@]}"; do
        if [ "$a" = "placeholder" ]; then out+=("$1"); else out+=("$a"); fi
    done
    printf '%s\n' "${out[@]}"
}

# ============================================================
# 7. 启动 Prefill（计算密集：大chunk、lpm前缀调度、radix cache开启）
# ============================================================
log ""
log ">>> [1/3] 启动 Prefill 实例（GPU $PREFILL_GPUS, TP=$PREFILL_TP）..."

mapfile -t PREFILL_ARGS < <(role_args prefill)
CUDA_VISIBLE_DEVICES="$PREFILL_GPUS" \
setsid "$PYTHON_BIN" -m sglang.launch_server \
    "${PREFILL_ARGS[@]}" \
    --port "$PREFILL_PORT" \
    --tp-size "$PREFILL_TP" \
    --mem-fraction-static "$PREFILL_MEM_FRACTION" \
    --max-running-requests "$PREFILL_MAX_RUNNING" \
    --schedule-policy "$SCHEDULE_POLICY" \
    --mamba-scheduler-strategy "$PREFILL_MAMBA_STRATEGY" \
    > "$LOG_DIR/prefill.log" 2>&1 &
echo $! > "$RUN_DIR/prefill.pid"
log "    Prefill PID: $(cat "$RUN_DIR/prefill.pid") | 日志: $LOG_DIR/prefill.log"

wait_ready prefill "$PREFILL_PORT" /health

# ============================================================
# 8. 启动 Decode（显存带宽密集：大batch、CUDA Graph覆盖到满并发）
#    注意：PD模式下SGLang会强制关闭decode侧radix cache，属正常行为。
#    此时 --max-running-requests 直接决定 mamba槽位数（=48，约3.6GB）。
# ============================================================
log ""
log ">>> [2/3] 启动 Decode 实例（GPU $DECODE_GPUS, TP=$DECODE_TP）..."

mapfile -t DECODE_ARGS < <(role_args decode)
CUDA_VISIBLE_DEVICES="$DECODE_GPUS" \
setsid "$PYTHON_BIN" -m sglang.launch_server \
    "${DECODE_ARGS[@]}" \
    --port "$DECODE_PORT" \
    --base-gpu-id 0 \
    --tp-size "$DECODE_TP" \
    --mem-fraction-static "$DECODE_MEM_FRACTION" \
    --max-running-requests "$DECODE_MAX_RUNNING" \
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
    > "$LOG_DIR/decode.log" 2>&1 &
echo $! > "$RUN_DIR/decode.pid"
log "    Decode PID: $(cat "$RUN_DIR/decode.pid") | 日志: $LOG_DIR/decode.log"

wait_ready decode "$DECODE_PORT" /health

# 回显引擎实际协商出的容量（这是判断参数是否生效的唯一依据）
log ""
log ">>> 引擎实际容量："
grep -hoE "max_total_num_tokens=[0-9]+.*" "$LOG_DIR/prefill.log" | tail -1 | sed 's/^/    prefill: /' || true
grep -hoE "max_total_num_tokens=[0-9]+.*" "$LOG_DIR/decode.log"  | tail -1 | sed 's/^/    decode : /' || true
grep -h "Mamba Cache is allocated" "$LOG_DIR/decode.log" | tail -1 | sed 's/^/    decode : /' || true

# ============================================================
# 9. 启动 Router（对外网关：限流、排队、重试、熔断、cache_aware路由）
# ============================================================
log ""
log ">>> [3/3] 启动 Router（对外端口 $ROUTER_PORT）..."

ROUTER_ARGS=(
    --pd-disaggregation
    --prefill "http://127.0.0.1:${PREFILL_PORT}" "$BOOTSTRAP_PORT"
    --decode  "http://127.0.0.1:${DECODE_PORT}"
    --host 0.0.0.0
    --port "$ROUTER_PORT"
    # 路由策略：prefill按前缀缓存亲和（提升prompt cache命中）
    --prefill-policy "$ROUTER_PREFILL_POLICY"
    # decode：单实例用round_robin；扩到>=2个decode实例时改成 power_of_two 做最小负载路由
    # （power_of_two 在只有1个decode worker时会直接拒绝启动）
    --decode-policy "$ROUTER_DECODE_POLICY"
    --cache-threshold "$ROUTER_CACHE_THRESHOLD"
    --max-tree-size "$ROUTER_MAX_TREE_SIZE"
    --eviction-interval-secs "$ROUTER_EVICTION_INTERVAL"
    # 限流与排队：超过并发上限先排队，队列满返回429，避免雪崩
    --max-concurrent-requests "$ROUTER_MAX_CONCURRENT"
    --queue-size "$ROUTER_QUEUE_SIZE"
    --queue-timeout-secs "$ROUTER_QUEUE_TIMEOUT"
    --request-timeout-secs "$REQUEST_TIMEOUT_SECS"
    --shutdown-grace-period-secs 180      # 优雅退出：等在途请求完成
    # 重试：流式响应已开始后重试会产生重复内容，生产上限制为2次
    --retry-max-retries 2
    --retry-initial-backoff-ms 100
    --retry-max-backoff-ms 5000
    # 熔断与健康检查
    --cb-failure-threshold 10
    --cb-timeout-duration-secs 60
    --health-check-interval-secs 30
    --health-check-timeout-secs 10
    --health-failure-threshold 3
    --worker-startup-timeout-secs 900
    # 可观测性
    --prometheus-host 0.0.0.0
    --prometheus-port "$PROMETHEUS_PORT"
    --log-dir "$LOG_DIR"
    --log-level info
    --max-payload-size 536870912          # 512MB，容纳多图/视频请求
)
[ -n "$SGLANG_API_KEY" ] && ROUTER_ARGS+=(--api-key "$SGLANG_API_KEY")

setsid "$PYTHON_BIN" -m sglang_router.launch_router "${ROUTER_ARGS[@]}" \
    > "$LOG_DIR/router.log" 2>&1 &
echo $! > "$RUN_DIR/router.pid"
log "    Router PID: $(cat "$RUN_DIR/router.pid") | 日志: $LOG_DIR/router.log"

wait_ready router "$ROUTER_PORT" /health

# ★ 关键：Router 的 /health 在 HTTP 服务起来的瞬间就返回200，但此时 worker 注册
#   还在后台进行（日志会打印 "Router ready | workers: []"）。此刻打请求必得 503
#   "No prefill workers available"。必须再等 /workers 里 prefill 和 decode 都健康。
log "    等待 worker 注册完成..."
for i in $(seq 1 120); do
    if curl -sf --max-time 5 "http://127.0.0.1:${ROUTER_PORT}/workers" 2>/dev/null \
        | "$PYTHON_BIN" -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
w=d.get('workers',[])
ok=lambda t: any(x.get('worker_type')==t and x.get('is_healthy') for x in w)
sys.exit(0 if (ok('prefill') and ok('decode')) else 1)
" 2>/dev/null; then
        log "    ✅ worker 注册完成（prefill + decode 均健康，耗时 ${i}s）"
        break
    fi
    [ "$i" -eq 120 ] && fail "worker 注册超时，请检查 $LOG_DIR/router.log"
    sleep 1
done

# ============================================================
# 10. 端到端冒烟测试（含 prompt cache 命中验证）
# ============================================================
AUTH_HEADER=()
[ -n "$SGLANG_API_KEY" ] && AUTH_HEADER=(-H "Authorization: Bearer $SGLANG_API_KEY")

if [ "$SKIP_SMOKE_TEST" != "1" ]; then
    log ""
    log ">>> 冒烟测试：端到端生成 + prompt cache 命中"
    SMOKE_BODY=$(cat <<EOF
{"model":"$SERVED_MODEL_NAME",
 "messages":[{"role":"system","content":"你是一个部署自检助手。请严格只回答被问到的内容。"},
             {"role":"user","content":"用一句话说明PD分离部署的作用。"}],
 "max_tokens":64,"temperature":0,
 "chat_template_kwargs":{"enable_thinking":false}}
EOF
)
    # 冒烟测试失败【不拆掉服务】：已经起好的实例比一次探测更值钱，
    # 只大声告警，由人工决定是否回滚。带重试以吸收worker刚注册后的抖动。
    SMOKE_OK=0
    for round in 1 2; do
        for attempt in 1 2 3; do
            RESP=$(curl -s --max-time 180 "http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions" \
                     -H 'Content-Type: application/json' "${AUTH_HEADER[@]}" \
                     -d "$SMOKE_BODY" 2>&1) || RESP=""
            if echo "$RESP" | "$PYTHON_BIN" -c "
import json,sys
r=json.load(sys.stdin)
if 'error' in r: sys.exit(1)
u=r['usage']; d=u.get('prompt_tokens_details') or {}
m=r['choices'][0]['message']; c=(m.get('content') or '').strip()
print(f\"    第${round}轮: prompt={u['prompt_tokens']} cached_tokens={d.get('cached_tokens')} completion={u['completion_tokens']} reasoning={len(m.get('reasoning_content') or '')}字\")
print('    输出:', c[:90].replace(chr(10),' '))
if not c: sys.exit(1)   # content为空说明生成或解析链路有问题
" 2>/dev/null; then
                SMOKE_OK=1; break
            fi
            log "    第${round}轮第${attempt}次未通过，5s后重试... ${RESP:0:200}"
            sleep 5
        done
    done
    if [ "$SMOKE_OK" = "1" ]; then
        log "    ✅ 冒烟测试通过（端到端生成正常）"
        log "    注：冒烟用的是短prompt(约39 tok)，短于复用粒度 ${CHUNKED_PREFILL_SIZE}，"
        log "        故 cached_tokens 为 0/None 属预期；且chat接口本身不回传该字段。"
        log "        验证prompt cache请用 >${CHUNKED_PREFILL_SIZE} tokens 的prompt，看 prefill.log 的 #cached-token"
    else
        log "    ⚠️⚠️  冒烟测试未通过，但服务仍在运行！请人工确认："
        log "         curl http://127.0.0.1:${ROUTER_PORT}/workers"
        log "         tail -50 $LOG_DIR/router.log"
    fi
fi

# ============================================================
# 11. 部署摘要
# ============================================================
cat <<EOF

============================================================
✅ Qwen3.8-27B PD分离部署完成（生产配置）
============================================================

服务拓扑
  Prefill : GPU $PREFILL_GPUS (TP=$PREFILL_TP) → http://127.0.0.1:${PREFILL_PORT}   并发上限 $PREFILL_MAX_RUNNING
  Decode  : GPU $DECODE_GPUS (TP=$DECODE_TP) → http://127.0.0.1:${DECODE_PORT}   并发上限 $DECODE_MAX_RUNNING
  （上面"引擎实际容量"一节打印的是协商后的真实值：prefill受 mamba槽位/3 约束，
    通常低于设定上限；这是SGLang对hybrid模型的硬约束，不是配置错误）
  Router  : http://0.0.0.0:${ROUTER_PORT}   ← 对外唯一入口
  指标    : http://0.0.0.0:${PROMETHEUS_PORT}/metrics

关键参数
  模型名       : $SERVED_MODEL_NAME
  上下文上限   : $MAX_MODEL_LEN tokens（input + max_tokens 之和不得超过此值）
  分块预填充   : $CHUNKED_PREFILL_SIZE
  网关并发/队列: $ROUTER_MAX_CONCURRENT / $ROUTER_QUEUE_SIZE（队列满返回 429）
  Prompt Cache : 生效中。复用粒度 = chunked_prefill_size = $CHUNKED_PREFILL_SIZE tokens
                 → prompt 短于 $CHUNKED_PREFILL_SIZE tokens 不会命中，这是预期行为
                 → 验证: grep "Prefill batch" $LOG_DIR/prefill.log | grep -o "#cached-token: [0-9]*"
  鉴权         : $([ -n "$SGLANG_API_KEY" ] && echo "Bearer <SGLANG_API_KEY>" || echo 未开启)

调用示例
  curl http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\$([ -n "$SGLANG_API_KEY" ] && printf "\n    -H 'Authorization: Bearer \$SGLANG_API_KEY' \\\\")
    -d '{"model": "$SERVED_MODEL_NAME",
         "messages": [{"role": "user", "content": "你好"}],
         "max_tokens": 1024,
         "chat_template_kwargs": {"enable_thinking": false}}'

  说明：模型默认开启思考模式，如需关闭请按上面传 chat_template_kwargs。

运维
  日志      : tail -f $LOG_DIR/{prefill,decode,router}.log
  PID       : $RUN_DIR/{prefill,decode,router}.pid
  优雅停止  : kill -TERM $$        （本脚本会按 router→decode→prefill 顺序停止）
  强制重启  : FORCE_RESTART=1 $0
  命中率监控: curl -s http://127.0.0.1:${PROMETHEUS_PORT}/metrics | grep -i cache

调优提示
  - 提高并发 : 加大 DECODE_MAX_RUNNING（每+1约需75.7MB显存）或 MAMBA_SSM_DTYPE=bfloat16（状态减半）
  - 加长上下文: MAX_MODEL_LEN=131072（KV按32KB/token计，单请求满长占4.2GB，并发会下降）
  - 压满显存 : DECODE_MEM_FRACTION 最高0.92，超过会因激活峰值OOM（实测激活约需2.4GB余量）
============================================================

EOF

# ============================================================
# 12. 守护：任一进程退出即整体退出（交给 systemd / supervisor 拉起）
# ============================================================
log ">>> 进入守护模式（Ctrl+C 停止全部服务）"
while true; do
    for svc in prefill decode router; do
        pid=$(cat "$RUN_DIR/$svc.pid" 2>/dev/null || true)
        if [ -z "${pid:-}" ] || ! kill -0 "$pid" 2>/dev/null; then
            log "❌ $svc 进程已退出，触发整体停止（请检查 $LOG_DIR/$svc.log）"
            exit 1
        fi
    done
    sleep 10
done
