// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to SGLang project

// Independent CUDA Driver API backend. No Mooncake dependency.
#include <cuda.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

static thread_local std::string error;
static void check(CUresult r) {
    if (r != CUDA_SUCCESS) {
        const char *s = nullptr; cuGetErrorString(r, &s);
        throw std::runtime_error(s ? s : "CUDA driver error");
    }
}
struct Region {
    CUmemGenericAllocationHandle handle = 0;
    CUdeviceptr ptr = 0;
    size_t bytes = 0, granularity = 0;
    bool mapped = false, poisoned = false;
    int device = 0;
    std::vector<CUcontext> contexts;
    void init(int selected) {
        check(cuInit(0));
        int n = 0; check(cuDeviceGetCount(&n));
        if (selected < 0 || selected >= n) throw std::invalid_argument("invalid local CUDA device");
        device = selected;
        for (int i = 0; i < n; ++i) {
            CUcontext ctx; check(cuDevicePrimaryCtxRetain(&ctx, i));
            contexts.push_back(ctx);
        }
    }
    void map(int tier, int location, bool owner) {
        CUmemAllocationProp actual{};
        check(cuMemGetAllocationPropertiesFromHandle(&actual, handle));
        auto expected = tier == 0 ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE;
        if (actual.location.type != expected ||
            (owner && !(actual.requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC)))
            throw std::runtime_error("allocation tier/FABRIC mismatch");
        check(cuMemAddressReserve(&ptr, bytes, granularity, 0, 0));
        check(cuMemMap(ptr, bytes, 0, handle, 0)); mapped = true;
        std::vector<CUmemAccessDesc> access;
        for (size_t i = 0; i < contexts.size(); ++i) {
            CUmemAccessDesc a{}; a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            a.location.id = i; a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            access.push_back(a);
        }
        if (owner && tier == 0) {
            CUmemAccessDesc a{}; a.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
            a.location.id = location; a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            access.push_back(a);
        }
        check(cuMemSetAccess(ptr, bytes, access.data(), access.size()));
    }
    void close() {
        if (poisoned) throw std::runtime_error("region quarantined after CUDA copy error; process restart required");
        if (mapped) { check(cuMemUnmap(ptr, bytes)); mapped = false; }
        if (ptr) { check(cuMemAddressFree(ptr, bytes)); ptr = 0; }
        if (handle) { check(cuMemRelease(handle)); handle = 0; }
        while (!contexts.empty()) {
            check(cuDevicePrimaryCtxRelease(contexts.size() - 1)); contexts.pop_back();
        }
    }
    ~Region() { try { close(); } catch (...) {} }
};
struct Context {
    explicit Context(Region *r) { check(cuCtxPushCurrent(r->contexts.at(r->device))); }
    ~Context() { CUcontext prev; cuCtxPopCurrent(&prev); }
};
#define BEGIN try {
#define END return 0; } catch (const std::exception &e) { error = e.what(); return -1; }
extern "C" {
int egm_abi_version() { return 3; }
const char *egm_error() { return error.c_str(); }
size_t egm_fabric_size() { return sizeof(CUmemFabricHandle); }
int egm_create(size_t n, int tier, int location, int device, void **out) { BEGIN
    if (!n || location < 0 || (tier != 0 && tier != 1)) throw std::invalid_argument("invalid size/tier/location");
    auto r = std::make_unique<Region>(); r->init(device);
    CUmemAllocationProp p{}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = tier == 0 ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = location;
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    check(cuMemGetAllocationGranularity(&r->granularity, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if (n > SIZE_MAX - r->granularity) throw std::invalid_argument("size overflow");
    r->bytes = (n + r->granularity - 1) / r->granularity * r->granularity;
    check(cuMemCreate(&r->handle, r->bytes, &p, 0)); r->map(tier, location, true);
    *out = r.release(); END }
int egm_import(const void *fabric, size_t n, size_t gran, int tier, int device, void **out) { BEGIN
    if (!n || !gran || n % gran || (tier != 0 && tier != 1)) throw std::invalid_argument("invalid imported size");
    auto r = std::make_unique<Region>(); r->init(device); r->bytes = n; r->granularity = gran;
    check(cuMemImportFromShareableHandle(&r->handle, const_cast<void *>(fabric), CU_MEM_HANDLE_TYPE_FABRIC));
    r->map(tier, 0, false); *out = r.release(); END }
int egm_export(void *v, void *fabric) { BEGIN
    auto r = static_cast<Region *>(v);
    check(cuMemExportToShareableHandle(fabric, r->handle, CU_MEM_HANDLE_TYPE_FABRIC, 0)); END }
size_t egm_size(void *v) { return static_cast<Region *>(v)->bytes; }
size_t egm_granularity(void *v) { return static_cast<Region *>(v)->granularity; }
// direction: 0 CPU->region, 1 region->CPU, 2 GPU->region, 3 region->GPU.
// Host return is a drain barrier, not just successful submission.
int egm_copy(void *v, size_t offset, uint64_t endpoint, size_t n, int direction) { BEGIN
    auto r = static_cast<Region *>(v);
    if (r->poisoned) throw std::runtime_error("region quarantined");
    if (offset > r->bytes || n > r->bytes - offset || !endpoint || !n || direction < 0 || direction > 3)
        throw std::invalid_argument("invalid copy range/direction");
    Context ctx(r);
    CUresult rc;
    if (direction == 0) rc = cuMemcpyHtoD(r->ptr + offset, reinterpret_cast<void *>(endpoint), n);
    else if (direction == 1) rc = cuMemcpyDtoH(reinterpret_cast<void *>(endpoint), r->ptr + offset, n);
    else if (direction == 2) rc = cuMemcpyDtoD(r->ptr + offset, endpoint, n);
    else rc = cuMemcpyDtoD(endpoint, r->ptr + offset, n);
    CUresult drained = cuCtxSynchronize();
    if (rc != CUDA_SUCCESS || drained != CUDA_SUCCESS) r->poisoned = true;
    check(rc); check(drained); END }
#ifdef EGM_WITH_GATHER
extern "C" int egm_gather_launch(const uint64_t *src, const uint64_t *dst, const uint64_t *size, uint64_t count,
                                 void *stream);
#endif
// Batch form of egm_copy for GPU<->region (direction 2/3); drains once. method: 0 one cuMemcpyDtoDAsync per
// run, ranges contiguous on both sides merged first; 1 one cuMemcpyBatchAsync; 2 one gather kernel
// (needs EGM_WITH_GATHER). *submitted = copy operations issued (runs for 0, ranges for 1/2).
int egm_copy_batch(void *v, size_t count, const size_t *offsets, const uint64_t *endpoints,
                   const size_t *sizes, int direction, int method, size_t *submitted) { BEGIN
    auto r = static_cast<Region *>(v);
    if (r->poisoned) throw std::runtime_error("region quarantined");
    if (direction != 2 && direction != 3) throw std::invalid_argument("batch copy supports GPU<->region only");
    if (method < 0 || method > 2) throw std::invalid_argument("invalid batch copy method");
#ifndef EGM_WITH_GATHER
    if (method == 2) throw std::invalid_argument("libegm built without the gather kernel");
#endif
    for (size_t i = 0; i < count; ++i)
        if (offsets[i] > r->bytes || sizes[i] > r->bytes - offsets[i] || !endpoints[i] || !sizes[i])
            throw std::invalid_argument("invalid copy range");
    std::vector<CUdeviceptr> dst, src;
    std::vector<size_t> len;
    if (method == 0) {
        std::vector<size_t> order(count);
        for (size_t i = 0; i < count; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return endpoints[a] < endpoints[b]; });
        for (size_t i : order) {
            CUdeviceptr region = r->ptr + offsets[i], ep = endpoints[i];
            CUdeviceptr d = direction == 2 ? region : ep, s = direction == 2 ? ep : region;
            if (!len.empty() && dst.back() + len.back() == d && src.back() + len.back() == s) len.back() += sizes[i];
            else { dst.push_back(d); src.push_back(s); len.push_back(sizes[i]); }
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            CUdeviceptr region = r->ptr + offsets[i], ep = endpoints[i];
            dst.push_back(direction == 2 ? region : ep); src.push_back(direction == 2 ? ep : region); len.push_back(sizes[i]);
        }
    }
    Context ctx(r);
    CUstream stream; check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CUresult rc = CUDA_SUCCESS;
    CUdeviceptr meta = 0;
    if (method == 0) {
        for (size_t i = 0; i < len.size() && rc == CUDA_SUCCESS; ++i)
            rc = cuMemcpyDtoDAsync(dst[i], src[i], len[i], stream);
    } else if (method == 1) {
        CUmemcpyAttributes attr{}; attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
        size_t attr_index = 0;
        rc = cuMemcpyBatchAsync(dst.data(), src.data(), len.data(), len.size(), &attr, &attr_index, 1, stream);
    } else {
#ifdef EGM_WITH_GATHER
        std::vector<uint64_t> host(3 * count);
        for (size_t i = 0; i < count; ++i) { host[i] = src[i]; host[count + i] = dst[i]; host[2 * count + i] = len[i]; }
        rc = cuMemAlloc(&meta, host.size() * sizeof(uint64_t));
        if (rc == CUDA_SUCCESS) rc = cuMemcpyHtoDAsync(meta, host.data(), host.size() * sizeof(uint64_t), stream);
        if (rc == CUDA_SUCCESS) {
            auto m = reinterpret_cast<const uint64_t *>(meta);
            rc = static_cast<CUresult>(egm_gather_launch(m, m + count, m + 2 * count, count, stream));
        }
#endif
    }
    CUresult drained = cuStreamSynchronize(stream);
    if (meta) cuMemFree(meta);
    cuStreamDestroy(stream);
    if (rc != CUDA_SUCCESS || drained != CUDA_SUCCESS) r->poisoned = true;
    check(rc); check(drained);
    *submitted = len.size(); END }
// Asynchronous grouped copy GPU<->region (direction 2/3). Ranges are split into ngroups consecutive groups
// (group_ends[g] = exclusive end); each group is one launch on a private stream followed by one event, so a
// caller can consume group g (e.g. one layer) while later groups are still copying. method 1 = one
// cuMemcpyBatchAsync per group, 2 = one gather kernel per group (metadata uploaded once, stream-ordered).
// The batch must be released with egm_batch_free, which drains the stream; any copy error poisons the region.
struct Batch {
    Region *region = nullptr;
    CUstream stream = nullptr;
    CUdeviceptr meta = 0;
    std::vector<CUevent> events;
    bool failed = false;
};
int egm_batch_submit(void *v, size_t count, const size_t *offsets, const uint64_t *endpoints, const size_t *sizes,
                     int direction, int method, size_t ngroups, const size_t *group_ends, void **out) { BEGIN
    auto r = static_cast<Region *>(v);
    if (r->poisoned) throw std::runtime_error("region quarantined");
    if (direction != 2 && direction != 3) throw std::invalid_argument("batch copy supports GPU<->region only");
    if (method != 1 && method != 2) throw std::invalid_argument("async batch supports method 1 or 2");
#ifndef EGM_WITH_GATHER
    if (method == 2) throw std::invalid_argument("libegm built without the gather kernel");
#endif
    if (!count || !ngroups || group_ends[ngroups - 1] != count) throw std::invalid_argument("invalid groups");
    for (size_t g = 0, prev = 0; g < ngroups; prev = group_ends[g++])
        if (group_ends[g] < prev) throw std::invalid_argument("group ends must be nondecreasing");
    for (size_t i = 0; i < count; ++i)
        if (offsets[i] > r->bytes || sizes[i] > r->bytes - offsets[i] || !endpoints[i] || !sizes[i])
            throw std::invalid_argument("invalid copy range");
    std::vector<uint64_t> host(3 * count);
    for (size_t i = 0; i < count; ++i) {
        CUdeviceptr region = r->ptr + offsets[i], ep = endpoints[i];
        host[i] = direction == 2 ? ep : region;              // src
        host[count + i] = direction == 2 ? region : ep;      // dst
        host[2 * count + i] = sizes[i];
    }
    Context ctx(r);
    auto b = std::make_unique<Batch>(); b->region = r;
    check(cuStreamCreate(&b->stream, CU_STREAM_NON_BLOCKING));
    CUresult rc = CUDA_SUCCESS;
    if (method == 2) {
        rc = cuMemAllocAsync(&b->meta, host.size() * sizeof(uint64_t), b->stream);
        if (rc == CUDA_SUCCESS) rc = cuMemcpyHtoDAsync(b->meta, host.data(), host.size() * sizeof(uint64_t), b->stream);
    }
    for (size_t g = 0, begin = 0; g < ngroups && rc == CUDA_SUCCESS; begin = group_ends[g++]) {
        size_t n = group_ends[g] - begin;
        if (n) {
            if (method == 1) {
                CUmemcpyAttributes attr{}; attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
                size_t attr_index = 0;
                rc = cuMemcpyBatchAsync(reinterpret_cast<CUdeviceptr *>(&host[count + begin]),
                                        reinterpret_cast<CUdeviceptr *>(&host[begin]), &host[2 * count + begin],
                                        n, &attr, &attr_index, 1, b->stream);
            } else {
#ifdef EGM_WITH_GATHER
                auto m = reinterpret_cast<const uint64_t *>(b->meta);
                rc = static_cast<CUresult>(egm_gather_launch(m + begin, m + count + begin, m + 2 * count + begin, n,
                                                             b->stream));
#endif
            }
        }
        CUevent e = nullptr;
        if (rc == CUDA_SUCCESS) rc = cuEventCreate(&e, CU_EVENT_DISABLE_TIMING);
        if (rc == CUDA_SUCCESS) { b->events.push_back(e); rc = cuEventRecord(e, b->stream); }
    }
    if (rc != CUDA_SUCCESS) {
        // Submitted work may still be reading or writing: drain before reporting, then poison.
        cuStreamSynchronize(b->stream);
        r->poisoned = true;
        for (auto e : b->events) cuEventDestroy(e);
        if (b->meta) cuMemFree(b->meta);
        cuStreamDestroy(b->stream);
        check(rc);
    }
    *out = b.release(); END }
// Blocks until group g has landed.
int egm_batch_wait(void *v, size_t group) { BEGIN
    auto b = static_cast<Batch *>(v);
    if (group >= b->events.size()) throw std::invalid_argument("invalid group");
    Context ctx(b->region);
    CUresult rc = cuEventSynchronize(b->events[group]);
    if (rc != CUDA_SUCCESS) { b->failed = true; b->region->poisoned = true; }
    check(rc); END }
// 0 = every group landed, 1 = still copying.
int egm_batch_query(void *v, int *pending) { BEGIN
    auto b = static_cast<Batch *>(v);
    Context ctx(b->region);
    CUresult rc = cuEventQuery(b->events.back());
    if (rc == CUDA_ERROR_NOT_READY) { *pending = 1; return 0; }
    if (rc != CUDA_SUCCESS) { b->failed = true; b->region->poisoned = true; }
    check(rc); *pending = 0; END }
// Drains and releases the batch; reports a copy error once more so the caller cannot miss it.
int egm_batch_free(void *v) { BEGIN
    std::unique_ptr<Batch> b(static_cast<Batch *>(v));
    Context ctx(b->region);
    CUresult drained = cuStreamSynchronize(b->stream);
    if (b->meta) cuMemFreeAsync(b->meta, b->stream);
    cuStreamSynchronize(b->stream);
    for (auto e : b->events) cuEventDestroy(e);
    cuStreamDestroy(b->stream);
    if (drained != CUDA_SUCCESS || b->failed) b->region->poisoned = true;
    check(drained);
    if (b->failed) throw std::runtime_error("batch copy failed");
    END }
int egm_close(void *v) { BEGIN
    auto r = static_cast<Region *>(v); r->close(); delete r; END }
}
