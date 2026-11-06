# EGM L2 sharing 的 master service：参考实现、实测与对 KVCM 的建议

2026-11-05

## 1. 背景与问题

在一个 NVL72 域内，sglang2 已经通过 KVCM 做 L1 sharing：Engine 把 KV 事件上报给 KVCM，目标 Engine 向 KVCM 查询前缀在哪个 Engine 的 HBM 里，再直接经 NVLink 读取。

下一步是 **L2 sharing**：把 KV 写透到 Grace CPU 的内存（EGM，经 CUDA Fabric 导出后可被域内任意 GPU 直接读写），让域内所有 Engine 共享这一层容量。问题是：**这一层的 master（谁在哪、写到哪、何时淘汰）能不能由 KVCM 承担，需要 KVCM 做哪些扩展**。

为回答这个问题，我们写了一个最小的参考实现并在 GB200 NVL72 上实测。本文给出它的分工、协议、实测数据，以及据此对 KVCM 的扩展建议。

## 2. 参考实现的分工

三个角色：

```
master（每个 NVLink 域一个进程，只管元数据）
   ▲ 注册 / 心跳 / 上报增删            ▲ 查找 / 放置
   │                                  │
store 服务（每节点每 NUMA 一个）      Engine（sglang，经 external linker 接入）
  持有一块 EGM slab、分配槽位、          启动时导入各 store 的 slab（一次）
  Pin / 租约、LRU                       读：acquire(keys) → 每层一次 gather kernel → release
                                        写：alloc(keys) → 一次 gather kernel → commit
```

| | master | store 服务 | Engine |
|---|---|---|---|
| 内存与槽位 | 不碰 | 持有 slab、分配槽位、LRU 淘汰 | — |
| 读保护 | 不管 | Pin + 截止时间（读方发 RPC 前取本机时间 + TTL）| 每层拷贝落地后核对截止时间，过期就放弃并重算 |
| 目录 | key → 哪个 store | **权威**：提交、淘汰都上报 | 查找、写前去重各一次 RPC |
| 成员与存活 | 注册（含 slab epoch、NVLink 域 ID）、两段判定（10 s 疑似 / 30 s 下线）| 定期上报即心跳 | 按成员列表导入新 store；store 重启后重新导入 |
| 放置 | 选空闲最多、能装下的活跃 store | 报告「不淘汰就能分配」的字节数 | 先写本地不淘汰 → 写 master 选的 store → 最后才在本地淘汰 |

**关键设计：目录只是提示。** 读方拿到「某页在 store X」后，仍然在 store X 上按 key `acquire`，偏移由 store 在 Pin 之下当场返回。所以：

- 目录过时（页已被淘汰）只会让 `acquire` 报缺页，Engine 退回重算，**不会读到被复用的槽位**；
- master **不需要分配代数**，也**不需要持久化**：store 的任何上报失败都改为重新注册 + 全量快照，master 重启后一个心跳内就被各 store 重建；
- master 不在数据路径上：master 不可达时 Engine 退回「并行问所有 store」，请求不失败。

## 3. 协议（一行一个 JSON，TCP + 作业密钥）与 KVCM API 的对照

| 操作 | 做什么 | KVCM 里最接近的 | 差别 |
|---|---|---|---|
| `register` | store 加入：地址、节点、NUMA、NVLink 域 ID、slab epoch、容量；同一 store 以新 epoch 注册时作废它的旧键 | `EVENT_NODE_REGISTER` | 拒绝其它 NVLink 域的 store（跨域无法导入 Fabric 内存）|
| `report` | 增量 `add` / `delete`，或 `snapshot` 全量替换；附带空闲字节与对象数，同时就是心跳 | `ReportEvent`（BLOCK_ADD / BLOCK_DELETE / BLOCK_SNAPSHOT / HEARTBEAT）| 心跳里的容量被放置使用 |
| `lookup` | 给一串页 key，返回最长已有前缀、每页一个持有者，按调用方节点 / NUMA 优先；也可逐页返回 | `GetCacheLocation(QT_PREFIX_MATCH)` / `GetHostCacheState` | 不返回偏移 |
| `place` | 返回空闲最多、能装下的活跃 store（扣掉刚放出去的估算）| `StartWriteCache` 的选址部分 | 只选 store，不分配偏移、不建写会话（写会话在 store 上）|
| `members` | 成员列表（状态、epoch、空闲）| `GetClusterInfo` | — |

store 与 Engine 之间的操作（`acquire` / `release` / `alloc` / `commit` / `abort`）不经过 master。

## 4. 实测

平台：GB200 NVL72，两个节点（同一 NVLink 域）。模型 Qwen2-1.5B（MHA），page 64 token，每页一个 1.75 MiB 对象。

### 4.1 功能（两节点，读方只配置本节点 store，另一节点靠 master 发现）

| 场景 | 结果 |
|---|---|
| 发现 | 读方经 master 找到另一节点上的页，跨节点读回，尾部 token 的 logprob 与重算差 ≤ 0.03 |
| 写前去重 | 另一节点已有的页不再写本地 |
| 溢出 | 本地 store 只给 0.25 GiB：16384 token 前缀的 256 页中，146 页写本地（不淘汰）、110 页由 master 放到另一节点；读方从两个 store 拼回整段前缀，logprob 与重算一致 |
| store 重启 | 新 epoch 注册后旧键作废；Engine 不重启，下一轮成员刷新时重新导入，继续读写 |
| master 重启 | master 停掉期间查找退回并行问所有 store，请求不失败；master 重启后 store 一个心跳内补发快照，查找回到 master |

### 4.2 开销：master 不拖慢恢复

同一对节点、同一代码，只差开不开 master；源端 8 路并发负载下，目标端从 EGM 恢复的 p50 / p99（ms）：

| 前缀 | 不开 master（并行问 2 个 store）| 开 master |
|---|---|---|
| 2048 token | 16.2–16.6 / 25.5–27.9 | 16.4–16.5 / 24.8–24.9 |
| 16384 token | 21.5–21.8 / 33.4–35.7 | 21.6–21.7 / 31.0–32.0 |

一次 master 查找（256 页）p50 / p99 为 0.23 / 0.53 ms，已快于并行问 2 个 store（0.29–0.33 / 0.61–0.80 ms）。

> **关于绝对值**：上表两列都偏高。测量时 Engine 侧每条计时记录都会重新打开一次日志文件，日志放在网络文件系统上，每次约 0.6 ms，其中一次落在调度线程上。
> 之后已修正（日志文件只打开一次）。同一对节点上修正前后对比，恢复 p50 在 2048 token 时少约 2.3 ms，16384 token 时少约 1.5 ms。
> 两列的开销相同，所以「开 master 不拖慢恢复」的结论不变。master 查找的耗时是在写日志之前计的，不受影响；4.3 只测控制面，不经过这部分代码，也不受影响。

### 4.3 规模：一个 NVL72（最多 36 个 store）

两节点，store 与 master 用同一套代码但不带 slab（只测控制面），每个 store 预置 3.3 万页：

| store 数 | 目录中的 key | 低负载查找 256 页 p50 / p99（master / 并行问）| 满负载吞吐 次/s（master / 并行问）| CPU 核（master / store 合计）|
|---|---|---|---|---|
| 2 | 6.6 万 | 0.41 / 2.17；0.50 / 3.46 | 4504 / 7448 | 1.2 / 2.4 |
| 16 | 52 万 | 0.53 / 2.18；2.20 / 8.63 | 4537 / 4289 | 1.2 / 17.4 |
| 36 | 118 万 | 1.05 / 3.31；5.56 / 16.9 | 3563 / 2305 | 1.2 / 21.2 |

| 查找页数（前缀长度）| 36 个 store 时 p50（master / 并行问）|
|---|---|
| 256（16K）| 1.05 / 5.56 ms |
| 2048（128K）| 5.9 / 21.4 ms |
| 8192（512K）| 20.7 / 77.4 ms |

- **一个域用一个 master 足够**；store 数到 16 以上时并行问就不如 master。
- **成本主要按 key 计**：约 1.6–2.5 µs / key（JSON 编解码 + 逐 key 查表），目录约 600 B / key。长前缀（128K 以上）的查找在毫秒到十几毫秒，与恢复本身（16384 token 约 21 ms）同量级。
- **上报成本低**：约 5.1 万新 key/s 的写透上报只占 master 0.35 核，且不影响查找延迟。

（参考实现是单进程 Python，以上是它的数字，不代表 C++ 实现的上限。）

## 5. 对 KVCM 的建议

按「改动小、收益大」排序。KVCM 现状依据 `alibaba/tair-kvcache@fadfe0f`（2026-11-05）的源码与仓库自带基准。

### 5.1 第 1 步：上报模式当 EGM 目录（KVCM 原则上不用改协议）

- KVCM 已有 reporter 型 L2：`ST_EVENT_REPORT_L2`（节点自己持有缓存、上报给 KVCM，其他节点 P2P 读），`GetHostCacheState` 已能给出各 host 的本地 / 全局联合前缀命中——与第 2 节的分工一致：**EGM store 当 reporter，KVCM 当目录**。
- 需要确认：`ST_EVENT_REPORT_L2` 的 URI 能否承载 store 地址（KVCM 会追加 `s_v`）；`medium` 用一个新值（如 `egm`）区分。
- **延迟是门槛**：参考实现在 36 个 store 时 256 块查找 p50 约 1 ms；KVCM 仓库自带的 Release 基准是 `GetHostCacheState` 100 块串行 0.68 ms、16 路并发 p50 5.3 ms（HTTP）。希望 KVCM 给出 gRPC、256 / 2048 块、多客户端并发下的数字。
- **按 key 的成本决定长前缀能不能用**：128K–512K 前缀是 2048–8192 块。降法：key 用 64 位整数而不是字符串；前缀查询利用链式哈希（第 i 块的 key 已编码前 i 块），服务端二分即可，O(log P) 次查表；请求只传必要的 key。
- 事件量：写透下每块一条 ADD；参考实现里 5.1 万 key/s 只占 0.35 核。KVCM 仓库自带的 ReportEvent 基准约 0.06–0.1 ms / 事件（debug 构建、Redis 后端，不代表 Release），按此折算同样的事件量要 3–5 核，可能需要按前缀合并事件。

### 5.2 第 2 步：KVCM 决定写到哪（需要两处小改）

- **消费心跳里的容量**：现在 `HEARTBEAT.system_status` 只用于观测。放置需要每个 store 的「不淘汰可分配字节」，以及「放出即扣减」以免并发写入方挤到同一个 store。
- **放置要感知拓扑**：同一 NVLink 域（跨域导入 Fabric 内存会失败）、写入方 GPU 的 NUMA 优先；再按空闲。
- **建议只选 store，不选偏移**：参考实现里 master 只回「写到哪个 store」，偏移由 store 在预留时给出。若沿用 `StartWriteCache` 定地址，则要么 KVCM 的后端在 `Create()` 里同步调 store 分配（多一次跨进程往返，分配器落在 KVCM 关键路径上），要么 `FinishWriteCache` 采用客户端回传的位置（现在 `locations` 字段被忽略）。

### 5.3 第 3 步：KVCM 管淘汰（需要精确读保护）

- 只要 KVCM 会触发删除，读方就要精确保护。参考实现的做法是**读方始终在 store 上按 key acquire**，所以目录不需要代数；若 KVCM 让读方直接用 URI 里的偏移读，则必须在 URI 里带分配代数（读前比对），或提供读租约（有未到期租约不删）。现在只有 `delay_before_delete_ms` 时间窗口（默认 0，内置启动配置 1000 ms），对 EGM 是正确性问题：槽位被复用后读方会读到别的页。
- 删除流程：KVCM 决定淘汰 → 通知 store 释放（store 仍按 Pin 拒绝正在读的）→ store 上报 DELETE。

### 5.4 不建议进 KVCM 的

数据面（Fabric 导入、gather kernel）与 slab 内的槽位分配：它们在每次读写的热路径上，放在 store 与客户端更快，也与 KVCM「不持有数据内存」的定位一致。

## 6. 希望 KVCM 团队确认的问题

1. `ST_EVENT_REPORT_L2` 的 URI / medium 能否直接承载 EGM store（地址、NUMA、NVLink 域）。
2. gRPC 下 `GetCacheLocation` / `GetHostCacheState` 在 256 / 2048 / 8192 块、多客户端并发时的延迟。
3. 是否计划在 `CacheLocation` 里带分配代数或提供读租约；`FinishWriteCache` 是否会采用回传位置。
4. 心跳 `system_status` 能否被放置策略读取（容量、NUMA、NVLink 域）。

## 附：代码位置（本分支）

- master：`python/sglang/srt/mem_cache/storage/egm/master.py`
- store 服务（注册、上报、槽位、Pin）：`python/sglang/srt/mem_cache/storage/egm/service.py`
- Engine 侧（查找、放置、导入、读写）：`python/sglang/srt/mem_cache/storage/egm/linker.py`
- 两节点示例（不起 master）：本目录的 `README.md`，`BACKEND=egm`
