# L1 KV 共享（SlotGuard）上游化整理进度

分支 `l1up/slotguard-glm`（sglang fork）。日期：2026-11-17。

这个分支把当前整理好的提交和它依赖的 PR 拼在一起，基于 main `2a71304ac5`（#48820 当前 head 所在的基点），可直接运行完整的 GLM-5.2 NVFP4 TP4 2P1D 远端 L1 恢复。
它用于分享进度，本身不是待合入的 PR。将来上游提交的是下面「P4 分层」里的各层；运行方法见 `test/manual/slotguard_linker/README.md`。

SlotGuard 的源端保护、控制面与目标端实现源自 hzh0425/sglang2 `nvlink/kv-sharing-opt`，本分支在此基础上整理。

## 1. 分支组成

自底向上：

| 段 | 提交 | 内容 |
|---|---|---|
| 基点 | `2a71304ac5` | sglang main |
| 合并 P1 | `36f8c98881` | #49474 peer reader（`7abeda232c`） |
| 合并 P3 | `5d0a6c8bf0` | #48820（`cfc8f87647`）+ P3‴ 请求级 prefetch 生命周期（`d1ba041e1f`，即 #50217） |
| P4 分层 | `cb2db6ae2f` … `6c34f37320` | 8 个提交，见第 3 节 |
| local | `cb873ed52a` | `test/manual/` 部署 harness（GLM gate、TTFT 驱动），不进上游 |
| 合并 indexer | `b20adb8b7b` | #44460（`af2f07fbd3`）+ #50216 的 P2#3–#5（`5c27613dd1`）。GLM 部署用的 indexer/bridge 从这里编译；`rust/sglang-server/src/api_server/core/handle.rs` 按 main 保持删除，`.github/workflows/pr-test-sgl-router.yml` 取 main 一侧 |
| 文档 | 本提交 | 本文件与 harness README 更新 |

## 2. 与现有 PR 的关系

状态为 2026-11-17 查询结果。

| 组成 | PR | 状态 | 本分支中的内容 |
|---|---|---|---|
| P1 peer reader | #49474 | open | `7abeda232c`，与 PR head 相同 |
| 异步 lookup | #48820 | open | `cfc8f87647`，与 PR head 相同 |
| P3‴ 请求级 prefetch 生命周期 | #50217 | draft | `d1ba041e1f`，与 PR head 相同 |
| KVCM bindings 与 golden（P2#1–#2） | #50216 的前两个提交 | draft | 在 P4d 段原样重放（`5b9ddf3217`、`88c3a04311`） |
| Tair KVCache indexer 后端 | #44460 | open | 由最后的合并引入，仅供 GLM 运行 |
| indexer 侧补充（P2#3–#5） | #50216 | draft | 同上 |
| SlotGuard 接入（P4 各层） | 未开 PR | — | 本地分层栈，第 3 节 |
| JIT VMM host 指针修复（P6） | #50212 | open | 不含：独立修复 |

## 3. P4 分层

每层一个提交（P4d 前有两个原样重放的 P2 提交）。行数是相对上一层的增删，「生产」指 `test/` 以外，「测试」指 `test/`。

| 层 | 提交 | 内容 | 生产 | 测试 |
|---|---|---|---|---|
| P4ab | `cb2db6ae2f` | CPU 页身份在分配、树、存储、linker 恢复路径上的透传；SlotGuard 账本；守护分页分配器 | +1827 / -26 | +1112 |
| P4c | `ed91ff6592` | 协议、线协议与回执、源端服务、TP 分配 epoch 组、catalog、发布 capture 及 URC / tree core 挂钩 | +6115 / -3 | +2090 / -1 |
| P4f | `132d81b096` | GLM DSA 的固定 FABRIC KV 池 | +300 | +402 |
| P4d | `5b9ddf3217`、`88c3a04311`（P2#1–#2）、`c2901e29e4` | KVCM bindings 与 golden；目标端传输、NVLink backend、linker transport（含远端读排队）、KVCM client | +734；+2768 | +7124 |
| Scheduler 拆分 | `bf69cea72b` | 在 KV 分配处拆开 prefill batch 构建，不带钩子；只依赖 main | +117 / -25 | — |
| P4e | `6c34f37320` | 通用 Scheduler / linker 扩展点、SlotGuard linker / runtime / config、configurator 选择分配器、能力拒绝条件 | +1869 / -5 | +3337 / -1 |

各层行数与 11-16 在 main `4f8b686753` 上的版本相同；迁到 `2a71304ac5` 时唯一的冲突是 P4ab 中 `static_kv_storage.py` 的一行 import。

整理期间做的结构调整：
- Scheduler 与 linker 的钩子改为通用扩展点：`CacheSchedulerHooks`、`prefetch_allocation_owner()`、`bind_tree_cache()`，通用代码不再 import SlotGuard 模块。
- 删除 V1 / lease 路径、V2 全 token Pin 和不可达开关；依赖部署 harness 的测试移到 `test/manual/`。
- 按调用方裁剪账本与分配器中没有调用方的路径，合并单测。

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

本分支对 main `2a71304ac5`：160 个文件，+42.5k / -127，其中：

| 来源 | 行数 | 说明 |
|---|---:|---|
| #49474 | +2.2k | 取代 sglang2 的 transport |
| #48820 | +0.2k / -45 | |
| P3‴ | +0.5k / -20 | |
| P4 各层 | +27.8k / -57 | 生产 12.9k（核心 1.9k、`l1_transfer` 10.3k、生成 0.6k），测试 14.8k |
| local harness | +3.8k | 不进上游 |
| #44460 + P2#3–#5 | +8.1k | 仅供 GLM 运行 |

对照：
- sglang2 去掉 `test/manual/` 与 sgl-router 后约 75.6k，对应的 P3‴ + P4 现为约 28.3k。
- 11-12 迁到新 main 时的扁平 P4 为 +37.2k（含 `test/manual/` 2.9k），当前 P4 各层合计 +27.8k，另有 harness 3.8k。

## 5. 测试

### 5.1 单测

镜像为 nightly 20261117（见 5.2）。下表的栈 `4f9afa37c8` 与本分支的 P4 各层相同，只差 local 层的 admin key 支持（见 5.3）。

| 范围 | 被测 | 结果 |
|---|---|---|
| 每个层前缀上「层号 ≤ 该层」的测试，P4e 起另加 Scheduler 相关上游用例 | 栈 `4f9afa37c8` 的 7 个前缀；local 层另在本分支的 `cb873ed52a` 上复跑 | 全部 0 failed（栈顶 608 passed） |
| 上游单测 `test/registered/unit/managers`、`test/registered/unit/mem_cache` | 栈顶 `4f9afa37c8`，对照 #48820 head `cfc8f87647` 与 P3‴ | 三棵树的失败集合逐条相同（各 123 failed、16 errors），没有新增失败 |
| CUDA（peer reader、peer engine、SlotGuard linker），GB300 | 栈顶 `4f9afa37c8` | 4 passed |

### 5.2 完整 GLM

- 环境：`lmsysorg/sglang:nightly-dev-cu13-20261117-25c1c439`（arm64 `sha256:763ce3cf6544d1928d6c79359da4fd3d06a116cd79cfcecf8b1527e9f7d4dcb6`），其中 sgl-deep-gemm 0.2.8、sgl-flash-mla 1.0.5、sglang-kernel 0.4.11 与基点 pyproject 一致。
- 部署：GB300，GLM-5.2 NVFP4，TP4，page64，2P1D 三节点，两个 Prefill 在同一 NVLink Fabric 域；P→D 走 Mooncake RDMA；Python TreeCore。
- 外部组件：indexer / bridge 从本分支编译；KVCM 为 Tair KVCache Manager `534df33b`。
- 两种驱动：
  - `DRIVER=gate`：开启字节校验与 trace，检查正确性；
  - `DRIVER=perf`：关闭两者，测 TTFT。`prefetch_policy` 都是 `wait_complete`。

**正确性 gate（本分支 `b20adb8b7b`）：通过**
- 4 次定向远端恢复（P0→P1、P1→P0，长度 257 与 1025），远端请求的 cached tokens 等于命中长度。
- 6 组确定性对照中，remote 与 source-hit、target-hit 与 source-hit 的输出全部一致。
- 10 个远端请求在 4 个 TP rank 上共 40 次字节校验，全部通过。
- 空闲时 pause / flush / continue 返回 200。

**cold 与命中的输出差异**：gate 中 cold（无缓存）与远端恢复的输出只有 1/4 一致，此前各版本也是如此。
我们在纯上游 main `4f8b686753` 上单节点、不开 L1 共享做了对照：同一实例上 cold 与本地命中的输出在 257、1025 长度也不一致，8257、32833 一致。
cold 与命中各自重复 3 次都可复现，`cached_tokens` 符合预期。

| 长度 | FP8 KV：cold = 命中 | BF16 KV：cold = 命中 |
|---|---|---|
| 257 | 0/6 | 0/6 |
| 1025 | 0/6 | 2/6 |
| 8257 | 6/6 | 6/6 |
| 32833 | 6/6 | 6/6 |

FP8 KV 下分别关闭 DSA 稠密 MHA（`SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0`）或 prefill CUDA graph，短 prompt 仍有不一致；根因尚未定位。
gate 的远端恢复都在 257、1025 两个长度，所以共享的正确性按 remote 与 source-hit 是否一致判断。

**TTFT 驱动的协议**：每轮先 flush 所有服务，然后依次发 5 个 lane，每个 lane C 个并发请求：

| lane | 做法 |
|---|---|
| source_cold | P0 冷 prefill 一组新 prompt |
| remote | 等 KVCM 列出 P0 后，把同一组 prompt 发到 P1，走远端恢复 |
| local_off | 另一组新 prompt 在 P1 自己 prefill |
| target_hit | 同一组 prompt 在 P1 再发一次，本地命中 |
| source_hit | 同一组 prompt 在 P0 再发一次，本地命中 |

**TTFT（本分支）**：每个场景跑 3 轮，单位为秒。perf 跑在加 admin key 之前的组合 `d3db280a22` 上，它与本分支只差 harness 的 4 个文件，`python/` 相同。

| 场景 | local_off | remote | target_hit | source_hit | remote/local_off | remote 全命中 | remote 与 source_hit 输出一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128K C8 | 24.42 | 1.58 | 1.11 | 1.15 | 0.064 | 24/24 | 24/24 |
| 256K C2 | 18.39 | 0.76 | 0.64 | 0.64 | 0.041 | 6/6 | 6/6 |
| 512K C2 | 46.52 | 1.82 | 1.81 | 1.84 | 0.039 | 6/6 | 6/6 |

11-16 在 main `4f8b686753`、nightly 20261113 镜像上的同一栈，三个场景的 remote/local_off 为 0.064、0.040、0.041，命中数和输出一致性相同。

**整理前后对比（11-16，main `4f8b686753`，未加远端读排队）**

对照臂是 11-12 迁移后、整理前的扁平 P4，补了同一行 PoolName 修正。两臂用同一镜像（nightly 20261113）、同一基点、同一份 indexer / KVCM。

| 场景 | 臂 | local_off | remote | remote/local_off | remote 全命中 | remote 与 source_hit 输出一致 |
|---|---|---:|---:|---:|---:|---:|
| 128K C8 | 整理前 | 23.81 | 19.64 | 0.825 | 4/24 | 24/24 |
| 128K C8 | 整理后 | 23.84 | 19.29 | 0.809 | 4/24 | 24/24 |
| 256K C2 | 整理前 | 18.38 | 12.39 | 0.674 | 3/6 | 6/6 |
| 256K C2 | 整理后 | 18.31 | 12.41 | 0.678 | 3/6 | 6/6 |
| 512K C2 | 整理前 | 46.35 | 31.63 | 0.682 | 3/6 | 6/6 |
| 512K C2 | 整理后 | 46.41 | 31.68 | 0.683 | 3/6 | 6/6 |

两臂的 gate 结果、TTFT、命中数和输出一致性均无差异。

### 5.3 整理与迁移中发现并修正的问题

- **PoolName 身份比较**：上游 #48013 让 `PoolTransfer.name` 规整为普通 `str`，SlotGuard linker 却用 `is not PoolName.KV` 判断，于是 GLM 上所有远端恢复都被拒、请求退回重算。改为值比较，并加了真实 `PoolTransfer` 的回归用例（P4e）。
- **并发远端恢复回退**：目标端同一时刻只允许一个远端读，这一点与 sglang2 相同；但 sglang2 用有界队列让其余请求依次进行，新 linker 却直接退回重算。现在在 P4d 的 linker transport 中按提交顺序排队，排队中的请求不占设备页（P4d）。
- **pause / continue 鉴权**：上游 #50559 起 `/pause_generation`、`/continue_generation` 要求服务端设置 admin API key，否则返回 403。harness 现在为 prefill 服务端生成 admin key，gate 在 flush / pause / continue 时携带（local 层）。

## 6. 尚未完成或未覆盖

- 读仍是串行的，并发更高时排队延迟线性增长。
- 同时未完成的 lookup 上限为 64。
- `prefetch_policy=timeout` 下，排队时间计入上游的 prefetch 超时；性能只在 `wait_complete` 下测过。
- 在途异常路径未验收：在途 abort / retract / flush、源端重启与旧 incarnation、真实 CUDA UNCERTAIN、单 rank 初始化失败。
- 短 prompt 的 cold 与命中输出差异在纯上游上也出现，根因未定位。
- 基点 `2a71304ac5` 比 2026-11-17 的 main `9158db2b97` 旧 56 个提交，尚未迁到最新 main。
- 各层仍有按 sglang 代码规范要处理的项，例如 dataclass 改 `msgspec.Struct`、去掉防御性 `getattr`；P4 各 PR 的拆分与提交方式未定。

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
