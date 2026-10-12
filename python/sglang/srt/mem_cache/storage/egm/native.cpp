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
int egm_abi_version() { return 2; }
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
int egm_close(void *v) { BEGIN
    auto r = static_cast<Region *>(v); r->close(); delete r; END }
}
