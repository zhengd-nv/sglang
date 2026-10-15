// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to SGLang project
// Adapted from SGLang storage/egm/native.cpp; see ../README.md for provenance.
// Modified: device-only VMM, per-device context, asynchronous range batches,
// event completion, active-region lifetime tracking and the nvt_* C ABI.
// Independent CUDA Driver data plane; no framework or transport-library dependency.
#include <cuda.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
void check(CUresult rc) {
    if (rc != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(rc, &msg);
        throw std::runtime_error(std::to_string(int(rc)) + ": " + (msg ? msg : "CUDA error"));
    }
}
struct Context {
    explicit Context(CUcontext ctx) { check(cuCtxPushCurrent(ctx)); }
    ~Context() { CUcontext old; cuCtxPopCurrent(&old); }
};
struct Region {
    CUcontext ctx = nullptr;
    CUdevice device = 0;
    CUmemGenericAllocationHandle allocation = 0;
    CUdeviceptr ptr = 0;
    size_t bytes = 0, logical = 0, granularity = 0;
    bool mapped = false, poisoned = false;
    unsigned active = 0;
    void init(int ordinal) {
        check(cuInit(0));
        check(cuDeviceGet(&device, ordinal));
        check(cuDevicePrimaryCtxRetain(&ctx, device));
    }
    void map() {
        Context scope(ctx);
        check(cuMemAddressReserve(&ptr, bytes, granularity, 0, 0));
        check(cuMemMap(ptr, bytes, 0, allocation, 0));
        mapped = true;
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        check(cuMemSetAccess(ptr, bytes, &access, 1));
    }
    void close() {
        if (active || poisoned) throw std::runtime_error("region busy or quarantined");
        if (ctx) {
            { Context scope(ctx);
              if (mapped) { check(cuMemUnmap(ptr, bytes)); mapped = false; }
              if (ptr) { check(cuMemAddressFree(ptr, bytes)); ptr = 0; }
              if (allocation) { check(cuMemRelease(allocation)); allocation = 0; }
            }
            check(cuDevicePrimaryCtxRelease(device)); ctx = nullptr;
        }
    }
    ~Region() { try { close(); } catch (...) {} }
};
void range(Region* r, size_t offset, size_t length) {
    if (!r || r->poisoned || !length || offset > r->logical || length > r->logical-offset)
        throw std::invalid_argument("invalid or quarantined region/range");
}
// state: 0 in flight, 1 success, 2 failed drained, 3 failed uncertain.
struct Batch {
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    CUevent event = nullptr;
    std::vector<Region*> regions;
    int state = 0;
    bool held = false;
    std::string failure;
    void finish(int result) {
        state = result;
        if (result == 3) for (auto* r : regions) r->poisoned = true;
        if (held && result != 3) {
            for (auto* r : regions) --r->active;
            held = false;
        }
    }
    void fail(CUresult cause) {
        const char* msg = nullptr; cuGetErrorString(cause, &msg);
        failure = msg ? msg : "CUDA transfer failure";
        CUresult drained = cuStreamSynchronize(stream);
        finish(drained == CUDA_SUCCESS ? 2 : 3);
    }
    void close() {
        if (state == 0 || state == 3) throw std::runtime_error("batch is not safely drained");
        Context scope(ctx);
        if (event) { check(cuEventDestroy(event)); event = nullptr; }
        if (stream) { check(cuStreamDestroy(stream)); stream = nullptr; }
    }
};
}
extern "C" {
struct NvRange { void* source; uint64_t source_offset; void* target; uint64_t target_offset; uint64_t length; };
int nvt_abi_version() { return 1; }
const char* nvt_error() { return last_error.c_str(); }
size_t nvt_fabric_size() { return sizeof(CUmemFabricHandle); }
#define BEGIN try {
#define END return 0; } catch (const std::exception& e) { last_error = e.what(); return -1; }
int nvt_create(int device, size_t length, void** out) { BEGIN
    if (!length || !out) throw std::invalid_argument("empty allocation");
    auto r = std::make_unique<Region>(); r->init(device); r->logical = length;
    CUmemAllocationProp prop{}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = r->device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    check(cuMemGetAllocationGranularity(&r->granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if (length > SIZE_MAX - r->granularity) throw std::invalid_argument("allocation overflow");
    r->bytes = ((length+r->granularity-1)/r->granularity)*r->granularity;
    { Context scope(r->ctx); check(cuMemCreate(&r->allocation, r->bytes, &prop, 0)); }
    r->map(); *out = r.release(); END }
// Trusted allocator contract: address is the BASE of one full VMM mapping,
// physical is that allocation's full size. Never accepts arbitrary tensor slices.
int nvt_register_vmm(int device, uint64_t address, size_t physical, void** out) { BEGIN
    if (!address || !physical || !out) throw std::invalid_argument("invalid VMM extent");
    auto r=std::make_unique<Region>(); r->init(device);
    { Context scope(r->ctx);
      check(cuMemRetainAllocationHandle(&r->allocation, reinterpret_cast<void*>(address)));
      CUmemAllocationProp prop{}; check(cuMemGetAllocationPropertiesFromHandle(&prop,r->allocation));
      if (prop.location.type!=CU_MEM_LOCATION_TYPE_DEVICE || prop.location.id!=r->device ||
          !(prop.requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC))
          throw std::invalid_argument("requires local DEVICE/FABRIC allocation");
      check(cuMemGetAllocationGranularity(&r->granularity,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    }
    if (physical%r->granularity) throw std::invalid_argument("unaligned VMM extent");
    r->logical=r->bytes=physical;
    // Own only a retained physical reference and a NEW alias VA, never the pool VA.
    r->map(); *out=r.release(); END }
int nvt_import(int device, const void* fabric, size_t length, size_t physical, size_t granularity, void** out) { BEGIN
    if (!fabric || !out || !length || length > physical || !granularity || physical%granularity)
        throw std::invalid_argument("invalid import descriptor");
    auto r = std::make_unique<Region>(); r->init(device);
    r->logical=length; r->bytes=physical; r->granularity=granularity;
    { Context scope(r->ctx);
      check(cuMemImportFromShareableHandle(&r->allocation, const_cast<void*>(fabric), CU_MEM_HANDLE_TYPE_FABRIC));
      CUmemAllocationProp prop{}; check(cuMemGetAllocationPropertiesFromHandle(&prop, r->allocation));
      if (prop.location.type != CU_MEM_LOCATION_TYPE_DEVICE) throw std::invalid_argument("only HBM DEVICE allocations supported");
    }
    r->map(); *out=r.release(); END }
int nvt_export(void* handle, void* fabric) { BEGIN
    auto* r=static_cast<Region*>(handle); if (!r || r->poisoned || r->active) throw std::runtime_error("region unavailable");
    Context scope(r->ctx);
    check(cuMemExportToShareableHandle(fabric,r->allocation,CU_MEM_HANDLE_TYPE_FABRIC,0)); END }
int nvt_info(void* handle, uint64_t* ptr, uint64_t* physical, uint64_t* granularity) { BEGIN
    auto* r=static_cast<Region*>(handle); *ptr=r->ptr; *physical=r->bytes; *granularity=r->granularity; END }
int nvt_host_copy(void* handle, size_t offset, void* data, size_t length, int read) { BEGIN
    auto* r=static_cast<Region*>(handle); range(r,offset,length);
    if (r->active) throw std::runtime_error("region in flight");
    Context scope(r->ctx);
    CUresult rc=read ? cuMemcpyDtoH(data,r->ptr+offset,length) : cuMemcpyHtoD(r->ptr+offset,data,length);
    CUresult drained=cuCtxSynchronize();
    if (rc!=CUDA_SUCCESS || drained!=CUDA_SUCCESS) r->poisoned=true;
    check(rc); check(drained); END }
int nvt_submit(const NvRange* ranges, size_t count, void** out) { BEGIN
    if (!ranges || !count || !out) throw std::invalid_argument("empty batch");
    auto b=std::make_unique<Batch>();
    for (size_t i=0;i<count;++i) {
        auto* s=static_cast<Region*>(ranges[i].source); auto* t=static_cast<Region*>(ranges[i].target);
        range(s,ranges[i].source_offset,ranges[i].length); range(t,ranges[i].target_offset,ranges[i].length);
        if (s->ctx!=t->ctx || (b->ctx && b->ctx!=s->ctx)) throw std::invalid_argument("mixed CUDA contexts");
        b->ctx=s->ctx; b->regions.push_back(s); b->regions.push_back(t);
    }
    Context scope(b->ctx);
    check(cuStreamCreate(&b->stream,CU_STREAM_NON_BLOCKING));
    CUresult rc=cuEventCreate(&b->event,CU_EVENT_DISABLE_TIMING);
    if (rc!=CUDA_SUCCESS) { cuStreamDestroy(b->stream); check(rc); }
    for (auto* r:b->regions) ++r->active;
    b->held=true;
    for (size_t i=0;i<count;++i) {
        auto* s=static_cast<Region*>(ranges[i].source); auto* t=static_cast<Region*>(ranges[i].target);
        rc=cuMemcpyDtoDAsync(t->ptr+ranges[i].target_offset,s->ptr+ranges[i].source_offset,ranges[i].length,b->stream);
        if (rc!=CUDA_SUCCESS) { b->fail(rc); break; }
    }
    if (b->state==0) { rc=cuEventRecord(b->event,b->stream); if(rc!=CUDA_SUCCESS) b->fail(rc); }
    *out=b.release(); END }
int nvt_poll(void* handle, int* state) { BEGIN
    auto* b=static_cast<Batch*>(handle); Context scope(b->ctx);
    if (b->state==0) {
        CUresult rc=cuEventQuery(b->event);
        if (rc==CUDA_SUCCESS) b->finish(1);
        else if(rc!=CUDA_ERROR_NOT_READY) b->fail(rc);
    }
    *state=b->state; END }
const char* nvt_batch_error(void* handle) { return static_cast<Batch*>(handle)->failure.c_str(); }
int nvt_release_batch(void* handle) { BEGIN auto* b=static_cast<Batch*>(handle); b->close(); delete b; END }
int nvt_close(void* handle) { BEGIN auto* r=static_cast<Region*>(handle); r->close(); delete r; END }
}
