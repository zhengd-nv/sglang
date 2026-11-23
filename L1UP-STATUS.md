# L1 KV 共享（SlotGuard）上游化整理进度

分支 `l1up/slotguard-glm`（sglang fork）。日期：2026-11-23。

这个分支把当前整理好的提交和它依赖的 PR 拼在一起，基于 main `7c21d8c701`，可直接运行完整的 GLM-5.2 NVFP4 TP4 2P1D 远端 L1 恢复。
它用于分享进度，本身不是待合入的 PR。将来上游提交的是下面「P4 分层」里的各层；运行方法见 `test/manual/slotguard_linker/README.md`。

SlotGuard 的源端保护、控制面与目标端实现源自 hzh0425/sglang2 `nvlink/kv-sharing-opt`，本分支在此基础上整理。各 PR 的署名见第 2 节。

## 1. 分支组成

自底向上：

| 段 | 提交 | 内容 |
|---|---|---|
| 基点 | `7c21d8c701` | sglang main |
| 合并 P1 | `4aaece8ae4` | [#49474](https://github.com/sgl-project/sglang/pull/49474) peer reader（`484796c428`） |
| 合并 P3 | `7a847330cb` | [#48820](https://github.com/sgl-project/sglang/pull/48820)（`93c0f82efa`）+ P3v4 请求级 prefetch 生命周期（`eaa3f40e81`，即 [#50217](https://github.com/sgl-project/sglang/pull/50217)） |
| P4 分层 | `756feec8e6` … `1b0eb44ac1` | 9 个提交，见第 3 节 |
| local | `2dc98284e7` | `test/manual/` 部署 harness（GLM gate、TTFT 驱动），不进上游 |
| 合并 indexer | `4cdf27e6f6` | [#44460](https://github.com/sgl-project/sglang/pull/44460)（`af2f07fbd3`）+ [#50216](https://github.com/sgl-project/sglang/pull/50216) 的 P2#3–#5（`5937df2a11`）。GLM 部署用的 indexer/bridge 从这里编译；`rust/sglang-server/src/api_server/core/handle.rs` 按 main 保持删除，`.github/workflows/pr-test-sgl-router.yml` 与 main 相同 |
| 文档 | 本提交 | 本文件与 harness README 更新 |

## 2. 与现有 PR 的关系

状态为 2026-11-23 查询结果。

| 组成 | PR | 状态 | 本分支中的内容 | `Co-authored-by: zhangheng` |
|---|---|---|---|---|
| P1 peer reader | [#49474](https://github.com/sgl-project/sglang/pull/49474) | open | `484796c428`，与 PR head 相同 | 不加：本方传输库，不来自 sglang2 |
| 异步 lookup | [#48820](https://github.com/sgl-project/sglang/pull/48820) | open | `93c0f82efa`，与 PR head 相同 | 不适用：他人的 PR |
| P3v4 请求级 prefetch 生命周期 | [#50217](https://github.com/sgl-project/sglang/pull/50217) | draft，叠在 #48820 上 | `eaa3f40e81`，与 PR head 相同 | 不加：新写，与 sglang2 逐字相同的生产行 2% |
| P4f 固定 FABRIC KV 池 | [#52439](https://github.com/sgl-project/sglang/pull/52439) | draft | 栈中 P4f，改动与提交说明都与 PR 提交 `bbf35f480f` 相同，基点不同 | 已加（2026-11-23 更新 PR 提交） |
| P4ab CPU 页身份与守护分页分配器 | [#52440](https://github.com/sgl-project/sglang/pull/52440) | draft，叠在 #52439 上 | 栈中 P4ab，改动与提交说明都与 PR 提交 `6e8c5ae764` 相同 | 已加（2026-11-23 更新 PR 提交） |
| P4c 线协议、P4h 源端服务、P4d 目标端传输、P4e 集成 | 未开 PR | — | 本地分层栈，第 3 节 | 已加在各层提交上 |
| KVCM bindings 与 golden（P2#1–#2） | [#50216](https://github.com/sgl-project/sglang/pull/50216) 的前两个提交 | draft | 在 P4d 前原样重放 | P2#1 已加；P2#2 是新写的 golden，不加（2026-11-23 更新 PR 提交） |
| Tair KVCache indexer 后端 | [#44460](https://github.com/sgl-project/sglang/pull/44460) | open | 由最后的合并引入，仅供 GLM 运行 | 不适用：他人的 PR |
| indexer 侧补充（P2#3–#5） | [#50216](https://github.com/sgl-project/sglang/pull/50216) | draft | 同上 | P2#3、P2#4 已加；P2#5 是新写的文档，不加 |
| JIT VMM host 指针修复（P6） | [#50212](https://github.com/sgl-project/sglang/pull/50212) | open | 不含：独立修复 | 不加：本方修复 |

署名依据（2026-11-23 实测）：对 sglang2 自分叉点起改过的文件逐行 `git blame`，再拿各层新增的非平凡行逐字匹配；改写过的行不计入，所以是下限。sglang2 上这些行的作者是 zhangheng（hzh0425）。

| PR | 生产行与 sglang2 逐字相同 | 测试行与 sglang2 逐字相同 |
|---|---|---|
| P4f #52439 | 76% | 35% |
| P4ab #52440 | 86% | 51% |
| P4c | 92% | 81% |
| P4h | 95% | 86% |
| P4d（含 P2#1–#2 重放） | 65% | 61% |
| P4e（含 Scheduler 拆分） | 55%（拆分提交 45%） | 64% |
| P2 #50216 | 生成绑定 97%；Rust indexer 侧 94%–96%；golden 与文档 0% | — |
| P3 #50217 | 2% | 0% |

## 3. P4 分层

每层一个提交（P4d 前有两个原样重放的 P2 提交）。行数是相对上一层的增删，「测试」指 `test/` 与 `python/sglang/test/`，「生产」指其余部分。

| 层 | 提交 | 内容 | 生产 | 测试 |
|---|---|---|---|---|
| P4f | `756feec8e6` | GLM DSA 的固定 FABRIC KV 池；拒绝会重新分配该池的组合（如 CUDA graph 持久化） | +297 | +406 |
| P4ab | `31d2b4bb33` | CPU 页身份（`PageSpan`）在分配、树、释放与 `PoolTransfer` 上的透传；SlotGuard 账本；守护分页分配器 | +1946 / -26 | +1392 |
| P4c | `eea13c8e6b` | 线协议：Pin/Unpin 格式与编解码、回执、hash-Pin 租约、directory、控制 profile、两端共用的传输 API | +2559 | +1485 |
| P4h | `9bdf2977b1` | 源端：Pin 服务与页导出、HTTP 控制端点、TP 分配 epoch 组、传输角色仲裁、catalog、发布 capture 及 URC / tree core 挂钩 | +3564 / -2 | +4692 / -1 |
| P4d | `f72983adcb`、`104f3d61c7`（P2#1–#2）、`dafe0d888b` | KVCM bindings 与 golden；目标端：基于 #49474 reader 的 kv_peer 引擎适配、NVLink backend、linker transport（含远端读排队）、KVCM client | +734；+2194 / -1 | +4116 |
| Scheduler 拆分 | `2950951f08` | 在 KV 分配处拆开 prefill batch 构建，不带钩子；只依赖 main | +105 / -24 | — |
| P4e | `1b0eb44ac1` | 通用 Scheduler / linker 扩展点、SlotGuard linker / runtime / config、configurator 选择分配器、能力拒绝条件 | +1859 / -12 | +3519 / -1 |

P4f、P4ab 已作为 #52439、#52440 打开；它们的 PR 分支直接基于 main，与栈中对应层的改动和提交说明相同。

整理期间做的结构调整：
- Scheduler 与 linker 的钩子改为通用扩展点：`CacheSchedulerHooks`、`prefetch_allocation_owner()`、`bind_tree_cache()`，通用代码不再 import SlotGuard 模块。
- 删除 V1 / lease 路径、V2 全 token Pin 和不可达开关；依赖部署 harness 的测试移到 `test/manual/`。
- 按调用方裁剪账本与分配器中没有调用方的路径，合并单测。
- 原先的 P4c 拆成线协议（P4c）和源端服务（P4h）两层。

## 4. 代码量

原始 sglang2（`cafe699fc` 对分叉点 `87211088c9`）：277 个文件，+116.1k / -113。

| 部分 | 行数 |
|---|---:|
| 核心 srt（`l1_transfer/` 以外） | +2.4k |
| `l1_transfer/`（其中 transport 1.6k） | +15.9k |
| 生成代码（KVCM bindings） | +0.6k |
| 测试（含 sgl-router Rust 测试） | +59.1k |
| `test/manual/` | +33.3k |
| 文档 | +2.1k |
| 其它（sgl-router 源码、脚本等） | +2.7k |

本分支（不含文档提交）对 main `7c21d8c701`：162 个文件，+43.0k / -129，其中：

| 来源 | 行数 | 说明 |
|---|---:|---|
| #49474 | +1.6k | 取代 sglang2 的 transport |
| #48820 | +0.2k / -47 | |
| P3v4 | +0.5k / -22 | |
| P4 各层 | +28.9k / -57 | 生产 13.2k（核心 2.0k、`l1_transfer` 10.5k、生成绑定与生成脚本 0.7k），测试 15.6k |
| local harness | +3.8k | 不进上游 |
| #44460 + P2#3–#5 | +8.1k / -25 | 仅供 GLM 运行 |

对照：
- sglang2 去掉 `test/manual/` 与 sgl-router 后约 75.6k，对应的 P3v4 + P4 现为约 29.4k。
- 11-17 在 main `2a71304ac5` 上的 P4 各层合计 +27.8k。

## 5. 测试

镜像为 `lmsysorg/sglang:nightly-dev-cu13-20261122-e96e9675`（arm64 `sha256:e5f39e93375cda37592c093fff705dc689b55845578aab4a1664f02b79d97c97`），其中 sgl-deep-gemm 0.2.9、sgl-deep-ep 0.1.3、sgl-flash-mla 1.0.5、sglang-kernel 0.4.11 与基点 pyproject 一致。单测与 GLM 都用 Python TreeCore（`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`）。

### 5.1 单测

| 范围 | 被测 | 结果 |
|---|---|---|
| 每个层前缀上「层号 ≤ 该层」的测试，P4e 起另加 Scheduler 相关上游用例 | P4f 至拆分提交在 11-22 的栈上测，P4e、local 在 11-23 的栈上测；这些提交与本分支对应层的树相同，local 只差 `test/manual/` 的 gate 判据（不在单测范围内） | 全部 0 failed（P4f 201 passed，P4e 608，local 630） |
| 上游单测 `test/registered/unit/managers`、`test/registered/unit/mem_cache` | 栈顶的较早版本 `485c466d88`（与本分支差 `config.py` 一处启动检查和 3 个测试文件），对照依赖组合 `7a847330cb` | 两树的失败集合逐条相同（各 509 failed、16 errors），没有新增失败。失败来自运行环境：镜像内的 Rust TreeCore 扩展与源码接口不符、CPU 节点没有 GPU、没有 cargo |
| CUDA（peer reader、peer engine、SlotGuard linker），GB300 | 同上 `485c466d88` | 16 passed。#49474 把 reader 的 GPU 用例并入 `test_kv_peer_read.py`，cuda 模式下该文件的 CPU 契约用例一起跑 |

### 5.2 完整 GLM

- 部署：GB300，GLM-5.2 NVFP4，TP4，page64，2P1D 三节点，两个 Prefill 在同一 NVLink Fabric 域；P→D 走 Mooncake RDMA。
- 外部组件：indexer / bridge 从本分支编译；KVCM 为 Tair KVCache Manager `534df33b`。
- 两种驱动：
  - `DRIVER=gate`：开启字节校验与 trace，检查正确性；
  - `DRIVER=perf`：关闭两者，测 TTFT。`prefetch_policy` 都是 `wait_complete`。

**正确性 gate：通过**（2026-11-23，nvl72d405-T[10-12]；被测组合 `fc0bb0aac6` 与本分支的树只差 harness README 的文字，提交信息另有署名更新）
- 4 次定向远端恢复（P0→P1、P1→P0，长度 257 与 1025），远端请求的 cached tokens 等于命中长度。
- 10 个远端请求在 4 个 TP rank 上共 40 次字节校验，全部通过。
- 6 组对照中，目标端 remote 与目标端自己命中的输出 6/6 一致；remote 与源端实例命中的输出 2/6 一致，只记录，原因见下文「跨实例的输出差异」。
- 空闲时 pause / flush / continue 返回 200。
- 4 次恢复中 cold 与 remote 的输出 1/4 一致，此前各版本也是如此，见下文「cold 与命中的输出差异」。

**跨实例的输出差异（新 main）**：在 `7c21d8c701` 上，gate 第一次按旧判据（remote 与源端实例上的命中输出一致）运行时，6 组只有 1 组一致；而目标端 remote 与目标端自己的命中 6/6 一致，字节校验全部通过。我们在纯上游 `7c21d8c701` 上起了两个配置相同的单节点实例（不开 L1 共享，FP8 KV），发同样的 prompt：

| 比较 | 结果 |
|---|---|
| 同一实例内重复 cold、重复命中 | 两个实例都是 9/9 一致 |
| 同一实例内 cold 与命中 | 3/9、4/9 一致（只有 8257 长度全部一致） |
| 两个实例之间同一请求 | warmup 不一致；cold 3/9 一致（只有 8257）；命中 4/9 一致 |

也就是说，在新 main 上，即使不涉及 L1 共享，两个实例对同一短 prompt 的输出也可能不同，但每个实例自身的输出是确定的。原因（镜像中的依赖版本、main 的改动或节点差异）尚未定位；旧基点 `2a71304ac5` 与 nightly 20261117 镜像上，remote 与源端命中一直一致。因此 gate 现在的判据是：逐 rank 字节校验全部通过，且在目标端同一实例上 remote 与命中输出一致；跨实例的比较只记录，不作为判据。

**cold 与命中的输出差异**（此前已有）：同一实例上 cold 与本地命中的输出在 257、1025 长度也不一致，8257、32833 一致。下表在纯上游 main `4f8b686753` 上单节点测得：

| 长度 | FP8 KV：cold = 命中 | BF16 KV：cold = 命中 |
|---|---|---|
| 257 | 0/6 | 0/6 |
| 1025 | 0/6 | 2/6 |
| 8257 | 6/6 | 6/6 |
| 32833 | 6/6 | 6/6 |

FP8 KV 下分别关闭 DSA 稠密 MHA（`SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0`）或 prefill CUDA graph，短 prompt 仍有不一致；根因尚未定位。

**TTFT 驱动的协议**：每轮先 flush 所有服务，然后依次发 5 个 lane，每个 lane C 个并发请求：

| lane | 做法 |
|---|---|
| source_cold | P0 冷 prefill 一组新 prompt |
| remote | 等 KVCM 列出 P0 后，把同一组 prompt 发到 P1，走远端恢复 |
| local_off | 另一组新 prompt 在 P1 自己 prefill |
| target_hit | 同一组 prompt 在 P1 再发一次，本地命中 |
| source_hit | 同一组 prompt 在 P0 再发一次，本地命中 |

**TTFT**：每个场景跑 3 轮，单位为秒。新基点一行是本分支的运行代码（2026-11-23，nvl72d008-T[01,03,05]），对照行是迁移前（2026-11-22，main `2a71304ac5`、nightly 20261117 镜像、当时的栈 `896c0d057e`）。

| 场景 | 基点 | source_cold | local_off | remote | target_hit | source_hit | remote/local_off | remote 全命中 | remote 与 target_hit 输出一致 | remote 与 source_hit 输出一致 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K C8 | `7c21d8c701` | 28.30 | 26.31 | 2.80 | 1.42 | 1.26 | 0.099 | 24/24 | 24/24 | 24/24 |
| 128K C8 | `7c21d8c701`，第 1–2 轮 | 23.85 | 24.17 | 1.27 | 1.15 | 1.16 | 0.053 | 16/16 | 16/16 | 16/16 |
| 128K C8 | 迁移前 | 24.07 | 23.89 | 1.52 | 1.14 | 1.12 | 0.064 | 24/24 | — | 24/24 |
| 256K C2 | `7c21d8c701` | 18.36 | 18.38 | 0.76 | 0.69 | 0.67 | 0.041 | 6/6 | 6/6 | 6/6 |
| 256K C2 | 迁移前 | 18.44 | 18.33 | 0.74 | 0.62 | 0.64 | 0.040 | 6/6 | — | 6/6 |
| 512K C2 | `7c21d8c701` | 46.50 | 46.59 | 1.99 | 1.99 | 1.99 | 0.043 | 6/6 | 6/6 | 6/6 |
| 512K C2 | 迁移前 | 46.57 | 46.29 | 1.81 | 1.79 | 1.84 | 0.039 | 6/6 | — | 6/6 |

- 新基点 128K C8 的第 0 轮所有 lane 都慢（source_cold 37.20、local_off 30.60、remote 5.87、target_hit 1.97），原因未查；第 1–2 轮与迁移前相当。
- 新基点 256K、512K 的三种命中 lane（remote、target_hit、source_hit）都比迁移前慢 0.03–0.20 s，remote 与本地命中同步变化。
- 128K 及以上的长 prompt 上，remote 与源端实例命中的输出全部一致；跨实例差异只在短 prompt 上出现。

### 5.3 512K 恢复的准备时间

在旧基点 `2a71304ac5`、nightly 20261117 镜像上，用迁移前的栈（`896c0d057e`）开 nvlink trace 测 512K C2：24 个目标 rank 视角的 transfer，每个 7999 页，中位数如下（毫秒）。

| 段 | 中位 |
|---|---:|
| KVCM lookup | 7.3 |
| Pin 往返 | 49.5 |
| 其中源端 catalog / guard / export | 8.8 / 6.4 / 1.2 |
| 其中源端组回复（重编码、自校验、构建） | 17.6 |
| TP 投票 | 5.6 |
| import | 8.8 |
| 拷贝 | 36.6 |
| `pin_start` 到 `group_ready` | 130 |

开 trace 时 remote TTFT 为 2.43 s，不开为 1.81 s，所以绝对值偏大。准备约占无 trace remote TTFT 的 7%；与 Pin 回复按页编码直接相关的部分（源端组回复，加上目标端解码与线上传输，后者约 15 ms、未细分）约占 1%–2%。当前实现按页编码。

### 5.4 整理与迁移中发现并修正的问题

- **PoolName 身份比较**：上游 #48013 让 `PoolTransfer.name` 规整为普通 `str`，SlotGuard linker 却用 `is not PoolName.KV` 判断，于是 GLM 上所有远端恢复都被拒、请求退回重算。改为值比较，并加了真实 `PoolTransfer` 的回归用例（P4e）。
- **并发远端恢复回退**：目标端同一时刻只允许一个远端读，这一点与 sglang2 相同；但 sglang2 用有界队列让其余请求依次进行，新 linker 却直接退回重算。现在在 P4d 的 linker transport 中按提交顺序排队，排队中的请求不占设备页（P4d）。
- **pause / continue 鉴权**：上游 #50559 起 `/pause_generation`、`/continue_generation` 要求服务端设置 admin API key，否则返回 403。harness 现在为 prefill 服务端生成 admin key，gate 在 flush / pause / continue 时携带（local 层）。
- **#49474 新 head 的接口**：reader 去掉了 `KvPeerExporter`、描述符中的 owner / region / lease 字段与 `close()`。P4d 的 kv_peer 引擎改用 `export_region()`，自己生成 owner id 与租约号（P4d）。
- **源端 `max_records`**：源端服务保留每个 transfer 的记录且不回收，`max_records` 因此是一个源端实例一生能服务的不同 transfer 数，超过后新 Pin 一律返回 BUSY，直到重启。P4e 现在从部署 manifest 读 `source_max_records`（默认 1024，范围 1 到 2^20）。
- **迁到 `7c21d8c701`**：
  - #51957 在 prefill 准备阶段决定 token 预算，Scheduler 拆分把它与混合 decode token 数一起传给分配阶段，TP4 入口投票改为比较五个标量（拆分提交、P4e）；
  - #52170 的采样池写入留在 `_finish_new_batch_prefill`（拆分提交）；
  - #51074 删除了 `ServerArgs.enable_dp_attention`，SlotGuard 的启动检查改读 `attn_dp_size`，这是 GLM 起服务时发现的（P4e）；
  - #42254 引入 CUDA graph 持久化（Foundry），它通过驱动钩子分配并重放引用 KV 池的 graph；固定 FABRIC 池没有对此测试，所以两者同时开启时启动即报错（P4f，#52439 同步）。
- **gate 判据**：见 5.2「跨实例的输出差异」（local 层）。

## 6. 尚未完成或未覆盖

- 读仍是串行的，并发更高时排队延迟线性增长。
- 同时未完成的 lookup 上限为 64。
- `prefetch_policy=timeout` 下，排队时间计入上游的 prefetch 超时；性能只在 `wait_complete` 下测过。
- 在途异常路径未验收：在途 abort / retract / flush、源端重启与旧 incarnation、真实 CUDA UNCERTAIN、单 rank 初始化失败。
- 源端 `max_records` 仍是实例一生的上限，只是改为可配置。
- 短 prompt 的 cold 与命中输出差异、新 main 上的跨实例输出差异，在纯上游上都能复现，根因未定位。
- 基点 `7c21d8c701` 比 2026-11-23 的 main `2be81e0ddc` 旧 7 个提交。
- 各层仍有按 sglang 代码规范要处理的项，例如 dataclass 改 `msgspec.Struct`、去掉防御性 `getattr`。P4c、P4h、P4d、P4e 尚未开 PR。

## 7. 运行

见 `test/manual/slotguard_linker/README.md`。要点：

```bash
# indexer / bridge（在 sgl-router 目录编译；workspace 依赖仓库顶层 rust/）
cd sgl-router && cargo build --release -p sgl-kv-indexer --bins
# CPU / CUDA 单测
bash test/manual/slotguard_linker/run-tests.sh cpu
bash test/manual/slotguard_linker/run-tests.sh cuda
# 三个节点分别执行；DRIVER=gate 为正确性 gate，DRIVER=perf 为 TTFT 驱动
DRIVER=gate bash test/manual/slotguard_linker/launch.sh <共享运行目录> p0|p1|d0
```
