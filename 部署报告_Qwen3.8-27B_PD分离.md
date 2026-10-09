# Qwen3.8-27B PD 分离部署报告

| 项目 | 内容 |
|---|---|
| 报告日期 | 2026-09-29 |
| 部署脚本 | `deploy_qwen3.8_pd.sh` |
| 状态 | ✅ 已上线运行，全部验证项通过 |
| 对外入口 | `http://<host>:19121`（OpenAI 兼容） |
| 对外模型名 | `Qwen3.8-27B` |

---

## 1. 环境

| 项 | 值 |
|---|---|
| GPU | 8 × NVIDIA A100-SXM4-**40GB**（本部署占用 GPU 2,3,4,5） |
| 互联 | 全对全 NV12 NVLink（`nvidia-smi topo -m` 确认），KV 传输走 NVLink 而非 IB |
| 驱动 | 580.178.04（支持 `cudaMemcpyBatchAsync`，是 INTRA_NODE_NVLINK 传输的前置条件） |
| 框架 | SGLang 0.5.10 + sglang_router |
| torch | 2.9.1+cu128 / CUDA 12.8 |
| Python 环境 | `sglang_env/.venv`（uv 管理） |

### 模型事实核对（与原脚本注释不符，已修正）

| 项 | 实际值 | 说明 |
|---|---|---|
| 权重精度 | **bfloat16** | 原脚本标题写 "FP8"，但 `config.json` 无 `quantization_config`；且 A100 (SM80) 无 FP8 算力 |
| 权重体积 | 52 GB（磁盘），TP=2 后 **25.64 GB/卡** | |
| 结构 | 64 层中仅 **16 层 full_attention**，48 层 linear_attention (Mamba) | 混合线性注意力，决定了本部署的全部容量特性 |
| 模型上限上下文 | 262144 | 但 40G 卡的 KV 池装不下，见下文 |
| 多模态 | `Qwen3_5ForConditionalGeneration`，支持图/视频 | |

---

## 2. 部署拓扑

```
                 客户端
                    │
                    ▼
        ┌───────────────────────────┐
        │  sglang_router  :19121    │  ← 唯一对外入口
        │  限流/排队/重试/熔断/路由  │     指标: :29000/metrics
        └───────┬───────────┬───────┘
                │           │
      cache_aware│           │round_robin
                ▼           ▼
   ┌──────────────────┐  ┌──────────────────┐
   │ Prefill :30000   │  │ Decode  :30001   │
   │ GPU 2,3  TP=2    │─▶│ GPU 4,5  TP=2    │
   │ 并发 12          │KV│ 并发 48          │
   └──────────────────┘  └──────────────────┘
            KV 经 NVLink 传输 (mooncake / INTRA_NODE_NVLINK)
            握手端口 :8998
```

引擎只监听 `127.0.0.1`，仅 Router 对外暴露。

---

## 3. 容量推导（这是选参数的依据，改参前必读）

### 3.1 单卡显存账本（40960 MiB）

```
权重                       25.64 GB   （TP=2 分片后）
CUDA/NCCL 上下文            ~1.70 GB
激活 / KV传输缓冲           ~2.40 GB   ← 在 mem-fraction-static 预算之外，实测得出
────────────────────────────────────
可用于 KV + Mamba 池        取决于 mem-fraction-static
```

原配置 `mem-fraction-static=0.80` 实测只用到 34.4 GB，**白白空出 6.86 GB**。
提到 0.86(prefill) / 0.90(decode) 后实测占用 36.4–37.4 GB，保留约 3.5 GB 余量，无 OOM。

### 3.2 两个关键单位成本（实测）

| 资源 | 单位成本 | 来源 |
|---|---|---|
| KV Cache | **32 KB / token**（每卡分片） | 16 full-attn 层 × 2(K,V) × 4 kv_head/2 × 256 head_dim × 2B |
| Mamba State | **75.7 MB / 请求** | 日志 `ssm_state 3.45GB / 48 槽` |

**Mamba State 是并发的真正瓶颈**，不是 KV。48 层线性注意力的状态必须整份常驻，
所以"能跑多少并发"几乎完全由 Mamba 槽位决定。

### 3.3 SGLang 对混合模型的并发硬约束（源码 `model_runner_kv_cache_mixin.py`）

```
开启 radix cache 时：max_running_requests = Mamba槽位 / 3     ← prefill 侧
关闭 radix cache 时：max_running_requests = Mamba槽位         ← decode 侧（PD 强制关闭）
```

这解释了为什么 prefill 并发上限设 16 而实际只有 12（= 38 槽 / 3）——
**这是框架硬约束，不是配置错误**。decode 侧显式设 48 会直接把 Mamba 槽位定为 48。

---

## 4. 最终生产参数

### 4.1 上下文与显存

| 参数 | 值 | 理由 |
|---|---|---|
| `--context-length` | **65536** | 输入+输出总上限，即客户端 `max_tokens` 天花板 |
| `--max-prefill-tokens` | 65536 | 必须 ≥ context-length，否则长 prompt 直接报错 |
| `--chunked-prefill-size` | **2048** | ★ 同时决定 prompt cache 复用粒度，见 §5 |
| `--mem-fraction-static` | 0.86 / 0.90 | prefill 激活峰值更大（视觉塔 + 不用 CUDA Graph），故留更多余量 |
| `--mamba-full-memory-ratio` | 0.6 | 偏向 KV，前缀缓存住在 KV 池里，池越大命中率越高 |

上下文长度权衡表（decode 侧 KV 池 176,556 tokens）：

| context-length | 单请求满长占 KV 池 | 可并发满长请求数 | 适用 |
|---|---|---|---|
| 32768 | 1.05 GB | ~19 | 高并发短对话 |
| **65536** | 2.10 GB | ~8 | ★ 默认，通用 |
| 131072 | 4.19 GB | ~4 | 长文档，并发明显下降 |
| 262144 | 8.39 GB | ~2 | 模型上限，不建议 40G 卡使用 |

### 4.2 并发与限流

| 层级 | 参数 | 值 | 行为 |
|---|---|---|---|
| Prefill 引擎 | `--max-running-requests` | 16（实际生效 12） | 受 Mamba槽位/3 约束 |
| Decode 引擎 | `--max-running-requests` | **48** | 直接决定 Mamba 槽位 = 48（3.45 GB） |
| Decode 引擎 | `--cuda-graph-max-bs` | 48 | 必须 ≥ 并发数，否则大 batch 退化为 eager |
| 引擎 | `--max-queued-requests` | 512 | 超出立即拒绝，不无限堆积 |
| Router | `--max-concurrent-requests` | 64 | 略高于 decode 并发，容纳传输中请求 |
| Router | `--queue-size` / `--queue-timeout-secs` | 512 / 300 | 队列满返回 **429** |

### 4.3 稳定性与可观测性

- **熔断**：`--cb-failure-threshold 10`，`--cb-timeout-duration-secs 60`
- **重试**：仅 2 次（流式响应开始后重试会产生重复内容，生产必须限制）
- **健康检查**：30s 间隔，3 次失败摘除
- **优雅退出**：`--shutdown-grace-period-secs 180`，等在途请求完成
- **看门狗**：`--watchdog-timeout 600`，单次 forward 卡死即自杀，由守护逻辑重启
- **指标**：引擎 `/metrics` + Router `:29000/metrics`，含 token 直方图
- **日志**：`--log-requests-level 0` 只记元数据，**不落盘 prompt 内容**

### 4.4 OpenAI 兼容层

| 参数 | 值 |
|---|---|
| `--served-model-name` | `Qwen3.8-27B`（原来对外暴露的是绝对路径，客户端不友好） |
| `--reasoning-parser` | `qwen3` → 思考内容分离到 `reasoning_content` |
| `--tool-call-parser` | `qwen3_coder` |
| `--limit-mm-data-per-request` | `{"image": 10, "video": 2}` 防止多模态 OOM |
| `--enable-cache-report` | 回传 `cached_tokens` |

---

## 5. Prompt Cache：本次最重要的发现

**结论：prompt cache 生效，但复用粒度 = `chunked_prefill_size`。**

prompt 长度低于该值时**完全不命中**，这一点非常反直觉，也是本次排查耗时最久的地方。

实测数据（同一前缀、后缀不同、连发两次）：

| chunked_prefill_size | prompt 3002 tok | 4802 tok | 6002 tok | 9002 tok | 16502 tok |
|---|---|---|---|---|---|
| 8192（初始） | cached 0 | 0 | 0 | 8192 | 8192 |
| **2048（最终）** | **2048** | 4096 | 4096 | 8192 | **16384** |

命中收益：3002 token 的重复前缀请求 **1.258s → 0.294s（快 4.3 倍）**。
最终配置下 `cache_hit_rate = 0.993`。

代价（8192 tok 唯一 prompt 冷压测）：

| chunked | 冷 TTFT (c=1) | 冷 TTFT (c=16) | 冷输出吞吐 |
|---|---|---|---|
| 8192 | 1226 ms | 16322 ms | 64.4 tok/s |
| 2048 | 1513 ms (+19%) | 17397 ms (+7%) | 59.8 tok/s (−7%) |

**取 2048**：冷启动慢 19%，但命中时快 4.3 倍。对 RAG / 长 system prompt / 多轮对话
这类真实负载净收益明显。若你的负载全是互不相同的长文档（无共享前缀），改回 8192 更快。

### ⚠️ 三个排查陷阱（已踩，勿重复）

1. **用完全相同的整条 prompt 连发两次 → `cached` 恒为 0**。
   末尾必须至少有 1 个新 token。真实业务的问题各不相同，天然满足；自测时请让后缀不同。
2. **Router 的 `/v1/chat/completions` 不返回 `prompt_tokens_details`**，
   看到 `None` 不代表没命中。权威信号只有两个：
   ```bash
   grep "Prefill batch" pd_logs/prefill.log | grep -o "#cached-token: [0-9]*"
   curl -s http://127.0.0.1:30000/metrics | grep cache_hit_rate
   ```
3. `cache_hit_rate` 是**累计平均值**，冷请求会拉低它，别只看一次。

### 被排除的假设（留档，避免重复劳动）

- ❌ 「PD 分离导致缓存失效」——在空闲 GPU 0,1 起单实例非 PD 对照组，行为一致
- ❌ 「`no_buffer` 策略导致」——改 `extra_buffer` 对命中率无改善，
  但 prefill 并发从 12 掉到 7（Mamba 槽位除数由 /3 变 /5），**纯亏，故不采用**

---

## 6. 修复的 4 个真实缺陷

| # | 缺陷 | 后果 | 修复 |
|---|---|---|---|
| 1 | 原脚本 cleanup 用全局 `pkill -f sglang.launch_server` | **会误杀同机其他模型部署**；实测中它反杀了刚启动的新实例 | 改为 `setsid` 进程组 + PID 文件 + 端口定向回收，并加自我保护（绝不杀自己所在进程组） |
| 2 | Router `/health` 假就绪 | HTTP 起来即返 200，但 worker 仍在后台注册（`workers: []`），此刻打请求必得 **503 No prefill workers available** | 增加 `/workers` 健康门禁，轮询至 prefill+decode 均 healthy |
| 3 | `--decode-policy power_of_two` | 该策略要求 ≥2 个 decode worker，单实例**直接拒绝启动** | 改 `round_robin`，并注明扩容到多实例后再切换 |
| 4 | 冒烟测试失败即 `exit` 触发 cleanup | 一次探测抖动就拆掉已正常的服务 | 改为重试 3 次 + **失败只告警不拆服务** |

另修正：GPU 注释与实际不符（注释写 GPU 0,1/2,3，实际 2,3/4,5）；标题 FP8 → bfloat16。

---

## 7. 验证结果

| 验证项 | 结果 |
|---|---|
| 启动流程 | prefill 30s → decode 35s → router 2s → worker 注册 2s，全程约 70s |
| 端到端生成 | ✅ 内容正确，`reasoning_content` 正常分离 |
| 长上下文 | ✅ 18,024 token prompt 通过（**原 16384 配置会直接拒绝**） |
| 并发 | ✅ 32 路并发 32/32 成功；64 路超并发 64/64 成功 |
| 显存 | 36.4–37.4 GB / 40.9 GB，余量约 3.5 GB，**无 OOM** |
| 请求回撤 | **0**（`total_retractions: 0`，无 `#retracted-req` 非零记录） |
| Prompt cache | ✅ `cache_hit_rate = 0.993` |
| 优雅停止 | ✅ `kill -TERM <脚本PID>` 按 router→decode→prefill 顺序停止，显存全部释放 |

---

## 8. 运维手册

```bash
# 启动（前台带守护）
./deploy_qwen3.8_pd.sh

# 后台启动（注意用 setsid，否则父 shell 被 kill 会带走整个部署）
setsid nohup ./deploy_qwen3.8_pd.sh > pd_deploy_run.log 2>&1 < /dev/null &

# 覆盖参数
MAX_MODEL_LEN=131072 ./deploy_qwen3.8_pd.sh
CHUNKED_PREFILL_SIZE=8192 ./deploy_qwen3.8_pd.sh    # 无共享前缀的负载
SGLANG_API_KEY=xxx ./deploy_qwen3.8_pd.sh           # 开启 Bearer 鉴权

# 端口被占时强制接管
FORCE_RESTART=1 ./deploy_qwen3.8_pd.sh

# 优雅停止
kill -TERM $(pgrep -f deploy_qwen3.8_pd.sh | head -1)

# 观测
tail -f pd_logs/{prefill,decode,router}.log
curl -s http://127.0.0.1:19121/workers | python3 -m json.tool
curl -s http://127.0.0.1:29000/metrics | grep -i cache
```

### 调用示例

```bash
curl http://127.0.0.1:19121/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3.8-27B",
       "messages": [{"role": "user", "content": "你好"}],
       "max_tokens": 1024,
       "chat_template_kwargs": {"enable_thinking": false}}'
```

> ⚠️ **本模型默认开启思考模式**。若 `max_tokens` 较小，全部 token 会进入
> `reasoning_content`，导致 `content` 为空。不需要思考链时必须传
> `chat_template_kwargs: {"enable_thinking": false}`。

---

## 9. 扩容与调优路线

| 目标 | 手段 | 代价 |
|---|---|---|
| 提高并发 | 加大 `DECODE_MAX_RUNNING`（每 +1 需 75.7 MB） | 挤占 KV 池 |
| 并发翻倍 | `MAMBA_SSM_DTYPE=bfloat16`（Mamba 状态减半） | **改变数值精度，须先离线评测** |
| 加长上下文 | `MAX_MODEL_LEN=131072` | 单请求满长占 4.19 GB，并发下降 |
| 压满显存 | `DECODE_MEM_FRACTION` 最高 0.92 | 超过会因激活峰值 OOM（需留 ~2.4 GB） |
| 提升 Router 路由质量 | 扩到 ≥2 个 decode 实例后切 `ROUTER_DECODE_POLICY=power_of_two` | 需更多 GPU |
| 更彻底的吞吐提升 | GPU 0,1,6,7 空闲，可扩成 4P+4D 或 2 组 2P+2D | — |

### 待跟踪

- SGLang 升级后复测 prompt cache 复用粒度是否解除 `chunked_prefill_size` 绑定
- `extra_buffer` 策略成熟后复测（可恢复 overlap 调度，当前并发代价过大）
