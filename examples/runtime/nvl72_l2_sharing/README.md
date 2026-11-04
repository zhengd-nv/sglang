# 在 NVL72 域内经 Grace 内存跨节点共享前缀 KV

这个示例在同一个 NVL72 域的两个节点上各起一个 SGLang 服务。**writer** 计算一段长前缀，
它的 KV 页经 write-through 写进一个 Mooncake store，store 放在其中一个节点的 Grace CPU 内存（EGM）里。
**reader** 从空的 GPU 缓存起步，在 store 里找到这段前缀，经 NVLink 把它读回来，而不是重算。
客户端检查两件事：reader 命中了缓存，并且它的输出与 writer 逐 token 一致。

```
 节点 0                                      节点 1（同一 NVL72 域）
 +---------------------------------+         +---------------------------+
 | mooncake_master + HTTP 元数据    |         |                           |
 | EGM provider（NUMA 0 Grace DRAM）|<------->| reader：sglang 服务        |
 | writer：sglang 服务（GPU 0）      | NVLink  |   （GPU 0）                |
 +---------------------------------+  fabric |   demo_client.py          |
                                             +---------------------------+
```

不经主机中转，也不走 RDMA：reader 的 GPU 经 NVLink fabric 直接读 provider 的 EGM 页。

这里的「节点」即 Slurm 节点，GB200 NVL72 上对应一个 compute tray（4 GPU + 2 Grace）；节点内按 NUMA 划分。

## 分支内容

| 提交 / 路径 | 作用 |
|---|---|
| `[HiCache] Direct linker pool group for plain MHA KV` | 让 direct external linker 支持普通 MHA 模型（上游 main 只覆盖 DSv4 / DSA / Mamba），拒绝 DCP 与打包的 draft KV |
| `[Feature] Unified Cache: Recompute requests whose external linker load fails` | store 读失败时撤回请求并重算（`--unified-cache-external-linker-load-failure`，默认 `recompute`），而不是返回 503 |
| `[HiCache] Direct linker: group layers per range read and keep reads in flight` | linker 按层读取时把几层合成一次调用（`load_layers_per_call`），并让多次调用同时在途（`load_inflight`）；默认 1 / 1 即原行为，本示例用 4 / 2 |
| `examples/runtime/nvl72_l2_sharing/` | 本示例 |
| `mooncake-nvlink-grouped-scatter.patch` | Mooncake 补丁：NVLink transport 把一次 scatter 作为一个 task 处理。不打的话每个 32 KiB 的页片段要单独轮询，负载下恢复 16k token 前缀约 70 ms 而不是 43 ms（不含上一行的 linker 改动）|

## 环境要求

- 两个 GB200 / GB300 节点在**同一个 NVL72 域**（`nvidia-smi -q | grep ClusterUUID` 相同），容器里有 IMEX channel
  （`/dev/nvidia-caps-imex-channels/`）。Slurm 的 block 拓扑集群上用 `--segment=2` 保证两个节点在一个域里。
- 本分支对应的 SGLang nightly 容器，例如 `lmsysorg/sglang:nightly-dev-cu13-20261102-38d9e4a3`；
  本分支装进去，或者把 `<repo>/python` 放在 `PYTHONPATH` 最前面。
- 一个两个节点都能看到的目录（下文的 `WORK`），放 Mooncake 构建与运行日志。
- 用 `Qwen/Qwen2-1.5B` TP1 测过（普通 MHA 模型都应该能用）。

## 1. 构建 Mooncake（一次，在容器里）

镜像自带的 Mooncake wheel（0.3.13.post1）没有 EGM store pool，所以要从源码构建。脚本固定 commit `522b716` 并打上 grouped-scatter 补丁：

```bash
WORK=/shared/l2-sharing ./build_mooncake.sh
```

它会写 `$WORK/env.sh`，其它脚本都会 source 它，把构建出的 `mooncake` 包排在镜像自带的之前。
如果镜像缺 yaml-cpp、zstd、xxHash 或 msgpack-c，用 Mooncake 的 `dependencies.sh` 装，或者传 `DEPS_PREFIX=<prefix>`。

然后让这份 checkout 能排在镜像的 SGLang 之前被导入（同样一次，在容器里）。脚本会拷贝镜像里已编好、但不在 git 里的 Rust 扩展；`run_slurm.sh` 会自动调用：

```bash
./prepare_sglang.sh
```

## 2. 运行

### 用 Slurm（两个节点一起）

```bash
srun -N2 --ntasks-per-node=1 --segment=2 --gpus-per-node=4 \
  --container-image=<sglang image> --container-mounts=/shared:/shared \
  env WORK=/shared/l2-sharing MODEL=Qwen/Qwen2-1.5B \
      PYTHONPATH=<this repo>/python \
  <this repo>/examples/runtime/nvl72_l2_sharing/run_slurm.sh
```

### 手动

```bash
# 每个节点：PYTHONPATH=<this repo>/python，先跑过 ./prepare_sglang.sh
# 节点 0
WORK=/shared/l2-sharing NUMAS=0 ./start_store.sh            # 等到 STORE_READY
WORK=/shared/l2-sharing MASTER_HOST=<节点 0 IP> MODEL=Qwen/Qwen2-1.5B GPU=0 ./start_server.sh
# 节点 1
WORK=/shared/l2-sharing MASTER_HOST=<节点 0 IP> MODEL=Qwen/Qwen2-1.5B GPU=0 ./start_server.sh
python3 demo_client.py --writer http://<节点 0 IP>:30000 --reader http://127.0.0.1:30000
```

先起 store 再起服务：每个服务启动时会做一次预热写入，这次写入需要已有 segment 可落。

`start_server.sh` 的 `LOAD_LAYERS_PER_CALL`（默认 4）与 `LOAD_INFLIGHT`（默认 2）控制按层读取的方式，见下文「实测」。

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

## 实测（GB200）

Qwen2-1.5B、TP1、page size 64、开 CUDA Graph、关 overlap。writer 上跑 8 路并发流式负载，reader 循环恢复前缀 60 s，
数值是负载下恢复延迟 p50 / p99（ms）。

**两节点，同一作业**（store 在节点 A；reader 在节点 B 读 A 的 EGM 为「跨节点」，在节点 A 读为「本节点」；linker 4 层一组 × 2 在途）：

| 前缀 | 跨节点 | 本节点 |
|---|---|---|
| 2048 | 15.0 / 17.8 | 14.8 / 17.4 |
| 16384 | 25.3 / 27.7 | 24.5 / 26.6 |

跨节点只比本节点多 0.2–0.8 ms；NVLink 计数显示数据只从 store 所在 NUMA 的入口 GPU 发到 reader GPU。
writer 开 write-through 后吞吐下降约 2%，写本节点与写另一节点没有可分辨的差别。

**单节点，与本地主机内存 L2（HiCache）对比**（同一作业，linker 4 层一组 × 4 在途）：

| 前缀 | 本示例 | HiCache 本地 host L2 |
|---|---|---|
| 2048 | 14.9 / 17.4 | 14.3 / 17.4 |
| 16384 | 26.5 / 29.1 | 17.5 / 20.9 |

**各项改动的贡献**（16384 前缀，负载下 p50，单节点，跨作业）：Mooncake 原版 70.0 ms → 打 grouped-scatter 补丁 42.9 ms → 再加 linker 分组与在途 24.5 ms。

说明：

- 短前缀的恢复与本地主机内存相当。长前缀仍慢约 7 ms，开销按页片段数而不是按字节数计：每页每层两个 32 KiB 片段，16384 token 就是 14336 个。
  linker 分组与在途已经把加载完全藏到 forward 后面；剩下的差距是：多个在途读调用里逐片段的驱动工作，会拖慢本服务自己的 kernel 发射。
  所以 `LOAD_INFLIGHT` 不宜超过 2（4 在途时加载更快，但整体反而慢 1–2 ms）。
- 不按层切分、整页读回时，微基准里 16384 token 只要约 7 ms。

两个节点上空闲时的一次运行（16384 token 前缀，生成 16 个 token，每个数值都含解码）：

| 请求 | 延迟 |
|---|---|
| writer，冷 prefill | 118 ms |
| writer，前缀在自己的 GPU 缓存里 | 58 ms |
| reader，前缀从另一个节点的 EGM 读回 | 66 ms（未加 linker 分组与在途时 83 ms）|

reader 所在节点只有 reader GPU 收到 939,523,072 字节，即预热 + 计时两轮各 448 MiB。

## 局限

- 只能在一个 NVL72 域内用。跨域时导入另一个节点的 fabric 内存会失败（`cuMemImportFromShareableHandle` 返回 800，not permitted），
  跨域请用走 RDMA 的 Mooncake。
- 只在 TP1、关 overlap 调度下验证过。TP > 1、PD 分离和投机解码都没测过。
- direct linker 不能和 `--enable-hierarchical-cache` 同时开，所以没有本地主机层：从 GPU 淘汰的内容只存在于 store 里。
- chunked prefill 遇到 store 读失败仍返回 503，而不是重算。

## 排查

| 现象 | 原因 |
|---|---|
| `start_store.sh` 报 `mooncake_master exited` | 该节点上元数据（18080）、RPC（50051）或 metrics（19003）端口被占。设 `METADATA_PORT` / `MASTER_PORT` / `METRICS_PORT`，`start_server.sh` 也设同样的 `METADATA_PORT` / `MASTER_PORT` |
| provider 报 `invalid argument` | 它看不到所在 NUMA 的入口 GPU。`PROVIDER_GPUS` 保持全部 GPU |
| provider 报 `EGM allocation failed ... CUDA result 2` | 该节点这块 NUMA 上空闲内存不够分配 EGM pool。先看 `/sys/devices/system/node/node*/meminfo`，换节点或调小 pool |
| 重启的服务报 `Duplicate rpc_meta key` | 上一个实例被杀时没有注销 transfer-engine 端点。换一个 `TE_PORT` 或重启 master |
| `flush_cache` 返回 400 | 上一个请求还在收尾，客户端会重试 |
| reader 的 `reader_cached_tokens` 明显小于前缀 | write-through 还没写完，调大 `--offload-wait` |
