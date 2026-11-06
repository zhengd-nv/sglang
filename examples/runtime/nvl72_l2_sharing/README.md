# 在 NVL72 域内经 Grace 内存跨节点共享前缀 KV

这个示例在同一个 NVL72 域的两个节点上各起一个 SGLang 服务。**writer** 计算一段长前缀，
它的 KV 页经 write-through 写进 Grace CPU 内存（EGM）里的一个 store。
**reader** 在另一个节点上，GPU 缓存是空的。它在 store 里找到这段前缀，经 NVLink 把它读回来，而不是重算。
客户端检查两件事：reader 命中了缓存，并且它的输出与 writer 逐 token 一致。

读写都不经主机中转，也不走 RDMA：GPU 经 NVLink fabric 直接读写另一个节点的 Grace 内存。

这里的「节点」即 Slurm 节点，GB200 NVL72 上对应一个 compute tray（4 GPU + 2 Grace）；节点内按 NUMA 划分。

## 两种后端

用 `BACKEND` 选择，两者用同一个客户端、同一套判据：

| | `mooncake`（默认）| `egm` |
|---|---|---|
| store | Mooncake Store：一个 master（带 HTTP 元数据服务）加每 NUMA 一个 EGM provider，只在节点 0 上 | 每个节点每 NUMA 一个 EGM store 服务（`sglang.srt.mem_cache.storage.egm.service`）|
| 写到哪 | master 分配（本示例只有节点 0 有 provider）| 本节点、本 NUMA 的 store |
| 怎么找到 | 问 master | 并行问所有 store，本节点优先 |
| 数据面 | Mooncake Transfer Engine 的 nvlink transport（`cudaMemcpyBatchAsync`）| 读方 / 写方 GPU 自己发 gather kernel，每层一次 |
| 依赖 | 从源码构建 Mooncake（带补丁），CMake 约 8 分钟 | CUDA driver、numpy；`nvcc` 编一个 `libegm.so`，几秒 |
| 端口 | master RPC、HTTP 元数据、metrics、每个进程一个 Transfer Engine 端口 | 每个 store 一个 TCP 端口（带作业密钥）|
| 全局能力 | master 统一分配、全局去重、高水位淘汰、跨节点借容量 | 跨 store 去重；无目录服务、不跨节点借容量 |

```
 BACKEND=mooncake                                      BACKEND=egm
 节点 0                          节点 1                 节点 0                          节点 1
 +-------------------------+     +----------+          +-------------------------+     +-------------------------+
 | master + 元数据          |     |          |          | EGM store（NUMA 0）      |     | EGM store（NUMA 0）      |
 | EGM provider（NUMA 0）   |<----| reader   |          | writer（GPU 0）写本节点   |<----| reader（GPU 0）读节点 0 |
 | writer（GPU 0）          | NVL | （GPU 0）|          |                         | NVL | 写前去重：不再写一份      |
 +-------------------------+     +----------+          +-------------------------+     +-------------------------+
```

## 读写过程

两种后端都挂在 SGLang 的 direct external linker 上（不开 HiCache，没有本地主机层）。SGLang 侧的流程相同，区别在 linker 怎么找、怎么读、往哪写。

### SGLang 侧（两种后端共用）

- **key**：radix cache 按页（64 token）算一条前缀哈希链，第 i 页的哈希覆盖前 i 页的全部 token。linker 在它前面加上模型标识，作为 store 里的 key，所以同一前缀在任何服务上得到同一组 key。
- **查找**：请求排进调度时，GPU 缓存没命中的部分交给 linker 的 `lookup`，返回从头开始连续命中的页数。只认连续前缀：中间缺一页，后面的页都不用。
- **读到 HBM**：命中的页先在本服务的 KV 池里分到一组私有页槽；linker 在后台线程里按层往这些槽里拷，每拷完一层就推进层计数器。forward 的第 l 层只等第 l 层的数据，所以加载与前面几层的计算重叠。forward 结束后这些页才插进 radix 树。
- **失败**：任何一步失败（key 不见了、租约过期、store 不可达、拷贝出错），请求被撤回、按整段重算（`--unified-cache-external-linker-load-failure recompute`，默认），不会把半读的 KV 当真。
- **写入时机**：`--hicache-write-policy write_through`：prefill 算出的新页在 forward 写完后交给 linker 的 `offload`，在后台线程里写；写完之前这些页在 radix 树里被锁住，不会被淘汰或覆盖。

### `BACKEND=mooncake`

**查找**：一次 `batch_is_exist` RPC 问 master：每页的 K、V 两个对象（`<页哈希>_<后缀>_k / _v`，每个对象含该页全部层）是否都在 store 里。

**读到 HBM**：
1. 对命中的全部 key 开一次读会话（`batch_get_session_start`），master 给这些对象加租约，期间不会被淘汰。
2. 按层组读：每次调用读 4 层（`LOAD_LAYERS_PER_CALL`），最多 2 个调用在途（`LOAD_INFLIGHT`）。每次调用 `batch_get_into_multi_buffer_ranges`，按 key 的副本位置找到 provider 的 EGM 段，
   每页每层的 K、V 各一个片段（Qwen2-1.5B 为 2 × 32 KiB），一起交给 Transfer Engine 的 nvlink transport，它用 `cudaMemcpyBatchAsync` 在本服务的 GPU 上从对方 EGM 直接拷进私有页槽。
3. 按层顺序推进层计数器；全部读完关会话（释放租约）。

**写入，确定目标**：offload 线程先 `batch_is_exist`，已存在的 key 跳过（全域按 key 去重），其余 `batch_put_from_multi_buffers`。
master 为每个对象挑 segment：先试请求带来的 `preferred_segments`（本示例没传），再从**所有已挂载、有空间的 segment 里随机挑**（默认 `--allocation_strategy random`），挑中的分配失败就顺序换下一个。
本示例只有节点 0 挂了 provider，所以全部写进节点 0 的 EGM；多个节点都挂 provider 时，写入会随机散到各节点。
对象先是「写入中」，写方经 nvlink 从自己的 GPU 拷进分到的位置，再通知 master 置为「完成」，之后才能被查到。

**写入，容量**：容量 = 各 provider 挂载的 EGM 之和（每 NUMA `POOL_GIB`，默认 64 GiB；SGLang 服务自己挂 0）。
master 后台在总使用率超过 90% 时按租约到期顺序淘汰约 5%，跳过有租约、未完成的对象。没有空间可分配时这次写入失败，这些页只是不进 store。

### `BACKEND=egm`

**查找**：并行问配置里的每个 store（一个 `exists` RPC 带上全部 key），每页取第一个持有它的 store，顺序是「本服务写入的 store 优先，其余按配置顺序」，并记下 key → store 供随后的读用。
对象是「一页一个」：该页全部层的 K，然后全部层的 V，与 SGLang MHA KV 池的布局一致。

**读到 HBM**：
1. 按 store 分组，每个 store 一次 `acquire`：整批 key 都在才加 Pin 并返回各自在 slab 里的偏移，缺任何一个就整批失败（走重算）。Pin 带截止时间：客户端在发 RPC 前取本机时间加 TTL。
2. 各 store 的 slab 在服务启动时已经导入过一次（FABRIC handle 经 `hello` 拿到），这里不再导入。linker 用 numpy 一次算出每页每层的源地址（slab 基址 + 偏移 + 层偏移）和目的地址（私有页槽），
   每层发一次 gather kernel（本服务的 GPU 直接从对方 EGM 读，经 NVLink），层后记一个 event。
3. 加载线程按层等 event、核对截止时间、推进层计数器；超时就不再推进，请求走重算。全部完成后 `release` 去 Pin。

**写入，确定目标**：
1. 跨 store 去重：并行问其它 store，已经在别处的页不写（所以本示例里 reader 读回的页不会再写一份）。
2. 其余页只写 `EGM_STORES` 的第一个，即本节点、本 GPU 所在 NUMA 的 store：写入只经过本 NUMA 的 C2C，不占别的节点的入口 GPU。**不会写到别的 store**。
3. `alloc`：store 对已存在或正被别人写的 key 回「已存在」，其余预留槽位并给出偏移与截止时间；linker 用一次 gather kernel 从 GPU 页拷进 slab，确认没过截止时间后 `commit`，store 这时才让别人查到。失败或超时则 `abort`。

**写入，容量**：每个 store 的容量在启动时固定（`POOL_GIB`，默认每 NUMA 64 GiB），按对象大小（64 B 对齐）切成槽位。
分配顺序：空闲表 → 未用过的空间 → **淘汰本 store 里最久未用、没被 Pin 的同尺寸对象**。都没有（全被 Pin）时这页回 FULL、不写。
过期或被放弃的预留先隔离 60 s 再复用，防止迟到的拷贝落进已被重新分配的槽位。
容量只在本 store 内回收，不会借别的节点：一个节点写得多，就在本地淘汰旧页，即使别的节点的 store 还空着（与 Mooncake 的差别见「局限」）。

## 分支内容

| 提交 / 路径 | 作用 |
|---|---|
| `[HiCache] Direct linker pool group for plain MHA KV` | 让 direct external linker 支持普通 MHA 模型（上游 main 只覆盖 DSv4 / DSA / Mamba），拒绝 DCP 与打包的 draft KV |
| `[Feature] Unified Cache: Recompute requests whose external linker load fails` | store 读失败时撤回请求并重算（`--unified-cache-external-linker-load-failure`，默认 `recompute`），而不是返回 503 |
| `[HiCache] Direct linker: group layers per range read and keep reads in flight` | Mooncake linker 按层读取时把几层合成一次调用（`load_layers_per_call`），并让多次调用同时在途（`load_inflight`）；默认 1 / 1 即原行为，本示例用 4 / 2 |
| `[HiCache] EGM store service and direct linker` | `python/sglang/srt/mem_cache/storage/egm/`：`egm` 后端的 store 服务、linker（经 `dynamic` linker 后端加载）与 native 库，见该目录的 README |
| `[HiCache] EGM: domain master for store membership, prefix directory and placement` 及其后两个修复 | `egm/master.py`：可选的域内 master（只管元数据：成员与存活、key 到 store 的目录、放置）；linker 配了 `master` 时一次 RPC 查找，本地 store 写满时溢出到 master 选的 store。本示例不起 master |
| `[HiCache] EGM linker: page objects over every KV-indexed pool, TP ranks` | 一页一个对象装全部以 KV 页为索引的池（MLA / DSA：latent KV + indexer K），支持 TP（复制池只有 rank 0 写，可选 rank 0 读后逐层广播）|
| `[HiCache] EGM linker: keep the audit file open` | linker 的计时日志只打开一次文件（每条重新打开在网络文件系统上约 0.6 ms，其中一条在调度线程上）|
| `egm-l2-master-reference.md` | EGM L2 sharing 的 master 参考实现：分工、协议、实测与对 KVCM 的扩展建议 |
| `examples/runtime/nvl72_l2_sharing/` | 本示例 |
| `mooncake-nvlink-grouped-scatter.patch` | Mooncake 补丁：NVLink transport 把一次 scatter 作为一个 task 处理。不打的话每个 32 KiB 的页片段要单独轮询，负载下恢复 16k token 前缀约 70 ms 而不是 43 ms（不含 linker 分组）|

## 环境要求

- 两个 GB200 / GB300 节点在**同一个 NVL72 域**（`nvidia-smi -q | grep ClusterUUID` 相同），容器里有 IMEX channel
  （`/dev/nvidia-caps-imex-channels/`）。Slurm 的 block 拓扑集群上用 `--segment=2` 保证两个节点在一个域里。
- 本分支对应的 SGLang nightly 容器，例如 `lmsysorg/sglang:nightly-dev-cu13-20261102-38d9e4a3`；
  本分支装进去，或者把 `<repo>/python` 放在 `PYTHONPATH` 最前面。
- 一个两个节点都能看到的目录（下文的 `WORK`），放构建产物、作业密钥与运行日志。
- 用 `Qwen/Qwen2-1.5B` TP1 测过（普通 MHA 模型都应该能用）。

## 1. 构建（一次，在容器里）

先让这份 checkout 能排在镜像的 SGLang 之前被导入。脚本会拷贝镜像里已编好、但不在 git 里的 Rust 扩展；`run_slurm.sh` 会自动调用：

```bash
./prepare_sglang.sh
```

**`BACKEND=egm`**：编 `libegm.so`，写 `$WORK/egm-env.sh`（`run_slurm.sh` 会自动调用）：

```bash
WORK=/shared/l2-sharing ./build_egm.sh
```

**`BACKEND=mooncake`**：镜像自带的 Mooncake wheel（0.3.13.post1）没有 EGM store pool，要从源码构建。脚本固定 commit `522b716` 并打上 grouped-scatter 补丁，
写 `$WORK/env.sh`，把构建出的 `mooncake` 包排在镜像自带的之前：

```bash
WORK=/shared/l2-sharing ./build_mooncake.sh
```

如果镜像缺 yaml-cpp、zstd、xxHash 或 msgpack-c，用 Mooncake 的 `dependencies.sh` 装，或者传 `DEPS_PREFIX=<prefix>`。

## 2. 运行

### 用 Slurm（两个节点一起）

```bash
srun -N2 --ntasks-per-node=1 --segment=2 --gpus-per-node=4 \
  --container-image=<sglang image> --container-mounts=/shared:/shared \
  env WORK=/shared/l2-sharing MODEL=Qwen/Qwen2-1.5B BACKEND=egm \
      PYTHONPATH=<this repo>/python \
  <this repo>/examples/runtime/nvl72_l2_sharing/run_slurm.sh
```

`BACKEND=egm` 时节点 0 生成作业密钥（`$WORK/runs/<job id>/egm.secret`，0600），两个节点各起本节点的 store，
再各起一个服务：每个服务的 store 列表是「本 NUMA 的 Unix socket 在前（写入目标），对方节点的 TCP 地址在后」。

### 手动（`BACKEND=egm`）

```bash
# 每个节点：PYTHONPATH=<this repo>/python，先跑过 ./prepare_sglang.sh 与 ./build_egm.sh
export WORK=/shared/l2-sharing SECRET_FILE=/shared/l2-sharing/egm.secret   # 两个节点用同一个密钥文件
NUMAS=0 ./start_egm_store.sh            # 每个节点各起一个，等到 STORE_READY

# 节点 0（writer）
BACKEND=egm MODEL=Qwen/Qwen2-1.5B GPU=0 \
  EGM_STORES=/tmp/egm-numa0.sock,tcp://<节点 1 IP>:29600 ./start_server.sh
# 节点 1（reader）
BACKEND=egm MODEL=Qwen/Qwen2-1.5B GPU=0 \
  EGM_STORES=/tmp/egm-numa0.sock,tcp://<节点 0 IP>:29600 ./start_server.sh
python3 demo_client.py --writer http://<节点 0 IP>:30000 --reader http://127.0.0.1:30000
```

两个节点的 store 都要先起来，再起服务：服务启动时会连上并导入列表里的每个 store。
`EGM_STORES` 的第一个是这个服务写入的 store，应是本节点、本 GPU 所在 NUMA 的那个（GPU 0/1 → NUMA 0，GPU 2/3 → NUMA 1）。

### 手动（`BACKEND=mooncake`）

```bash
# 节点 0
WORK=/shared/l2-sharing NUMAS=0 ./start_store.sh            # 等到 STORE_READY
WORK=/shared/l2-sharing MASTER_HOST=<节点 0 IP> MODEL=Qwen/Qwen2-1.5B GPU=0 ./start_server.sh
# 节点 1
WORK=/shared/l2-sharing MASTER_HOST=<节点 0 IP> MODEL=Qwen/Qwen2-1.5B GPU=0 ./start_server.sh
python3 demo_client.py --writer http://<节点 0 IP>:30000 --reader http://127.0.0.1:30000
```

先起 store 再起服务：每个服务启动时会做一次预热写入，这次写入需要已有 segment 可落。
`LOAD_LAYERS_PER_CALL`（默认 4）与 `LOAD_INFLIGHT`（默认 2）控制 Mooncake linker 按层读取的方式，见下文「实测」。

## 3. 看什么

`demo_client.py` 打印一份 JSON 汇总和一行结论：

```
DEMO_RESULT PASS prefix=16384 writer_cold=...ms reader_from_pool=...ms reader_cached_tokens=16384
```

- `reader_cached_tokens` 等于整个前缀：reader 没有重算。
- `reader output matches the writer's cached output`：读回来的就是 writer 的 KV。参照是 writer 从自己的 GPU 缓存作答，
  不是冷 prefill：冷 prefill 走的是另一条数值路径，在近似平局时可能合理地选出另一个 token。
- `run_slurm.sh` 还会打印 `READER_NODE_NVLINK_DELTA`，即演示期间 reader 节点上每块 GPU 的 NVLink 收发字节。
  只有 reader GPU 应该收到数据：每轮 `前缀 / page_size × 每页字节`（Qwen2-1.5B 每个 64 token 的页 1.75 MiB，16384 token 即 448 MiB），
  客户端会先跑一轮不计时的预热，再跑一轮计时的。
- `BACKEND=egm` 时还打印每个 store 的 `EGM_STORE ... {"objects": ...}`：节点 0 的 store 有 512 个对象
  （预热与计时两轮各一条前缀、各 256 页，共 896 MiB），节点 1 的 store 是 0：reader 读回后没有把前缀再写一份。

## 实测（GB200）

Qwen2-1.5B、TP1、page size 64、开 CUDA Graph、关 overlap。writer 上跑 8 路并发流式负载，reader 循环恢复前缀 60 s，
数值是负载下恢复延迟 p50 / p99（ms）。

**两节点，同一作业，两种后端**（store 在节点 A；reader 在节点 B 读 A 为「跨节点」，在 A 上读为「本节点」；Mooncake linker 4 层一组 × 2 在途）：

| 前缀 | egm 跨节点 | egm 本节点 | mooncake 跨节点 | mooncake 本节点 |
|---|---|---|---|---|
| 2048 | 16.3 / 22.1 | 15.8 / 22.9 | 15.0 / 17.8 | 14.8 / 17.4 |
| 16384 | 21.7 / 30.1 | 21.0 / 28.9 | 25.3 / 27.7 | 24.5 / 26.6 |

- 跨节点只比本节点多 0.2–0.8 ms；NVLink 计数显示数据只从 store 所在 NUMA 的入口 GPU 发到 reader GPU。
- 长前缀 `egm` 快约 3.6 ms：它每层一次 gather kernel，不受页片段数影响；Mooncake 按片段提交批量拷贝，片段多时驱动工作会拖慢本服务的 kernel 发射。
  短前缀 `mooncake` 快约 1.3 ms：`egm` 每次恢复多两次 RPC（查找要问所有 store，再加一次 Pin），p99 也更高。
- writer 开 write-through 后吞吐下降：`mooncake` 约 2%，`egm` 4–6%（offload 线程的 Python 与 RPC，不是拷贝）。写本节点与写另一节点没有可分辨的差别。

**单节点，与本地主机内存 L2（HiCache）对比**（同一作业；Mooncake linker 4 层一组 × 4 在途）：

| 前缀 | egm | mooncake | HiCache 本地 host L2 |
|---|---|---|---|
| 2048 | 16.1 / 28.9 | 14.9 / 17.4 | 14.3 / 17.4 |
| 16384 | 20.7 / 31.6 | 26.5 / 29.1 | 17.5 / 20.9 |

**Mooncake 各项改动的贡献**（16384 前缀，负载下 p50，单节点，跨作业）：原版 70.0 ms → grouped-scatter 补丁 42.9 ms → 再加 linker 分组与在途 24.5 ms。

两个节点上空闲时各一次运行（16384 token 前缀，生成 16 个 token，每个数值都含解码；两列是不同作业、不同节点对）：

| 请求 | mooncake | egm |
|---|---|---|
| writer，冷 prefill | 118 ms | 114 ms |
| writer，前缀在自己的 GPU 缓存里 | 58 ms | 58 ms |
| reader，前缀从另一个节点的 Grace 内存读回 | 66 ms | 65 ms |

## 局限

- 只能在一个 NVL72 域内用。跨域导入另一个节点的 fabric 内存会失败（CUDA 800，not permitted）；
  `egm` 在服务启动时比 `ClusterUUID` 直接拒绝，`mooncake` 在首次拷贝时报错。跨域请用走 RDMA 的 Mooncake。
- 只在 TP1、关 overlap 调度下验证过。TP > 1、PD 分离和投机解码都没测过；`egm` 只支持普通 MHA 的 KV 池。
- direct linker 不能和 `--enable-hierarchical-cache` 同时开，所以没有本地主机层：从 GPU 淘汰的内容只存在于 store 里。
- chunked prefill 遇到 store 读失败仍返回 503，而不是重算。
- `egm`：查找与写前去重每次都要问所有 store，适合几个节点；整个域（最多 36 个 store）需要目录服务或按 key 哈希路由。
  只写本节点的 store：满了在本地淘汰，不会借别的节点的容量（Mooncake 由 master 在全部 provider 里分配）。按页 LRU 可能先淘汰前缀的头几页。store 服务重启后，已运行的服务要一起重启。

## 排查

| 现象 | 原因 |
|---|---|
| `start_store.sh` 报 `mooncake_master exited` | 该节点上元数据（18080）、RPC（50051）或 metrics（19003）端口被占。设 `METADATA_PORT` / `MASTER_PORT` / `METRICS_PORT`，`start_server.sh` 也设同样的 `METADATA_PORT` / `MASTER_PORT` |
| provider 或 EGM store 报 `invalid argument` | 它看不到所在 NUMA 的入口 GPU。`PROVIDER_GPUS` 保持全部 GPU |
| provider 或 EGM store 报 EGM 分配失败（`CUDA result 2` / out of memory）| 该节点这块 NUMA 上空闲内存不够分配 pool。先看 `/sys/devices/system/node/node*/meminfo`，换节点或调小 `POOL_GIB` |
| 服务启动时报 `is in NVLink domain ..., this Engine in ...` | `egm`：列表里有别的 NVL72 域的 store。两个节点不在一个域，Slurm 上加 `--segment` |
| 服务启动时报 `bad or missing secret` | `egm`：两边用的不是同一个 `SECRET_FILE` |
| 重启的服务报 `Duplicate rpc_meta key` | `mooncake`：上一个实例被杀时没有注销 transfer-engine 端点。换一个 `TE_PORT` 或重启 master |
| `flush_cache` 返回 400 | 上一个请求还在收尾，客户端会重试 |
| reader 的 `reader_cached_tokens` 明显小于前缀 | write-through 还没写完，调大 `--offload-wait` |
