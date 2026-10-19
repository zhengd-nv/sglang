// Deterministic CUDA-call fault injection against the actual native.cpp state machine.
// These overrides model queued copies; they do not simulate a hardware reset.
// Run each scenario in a fresh process: quarantined resources intentionally survive.
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <algorithm>
#include "../../../python/sglang/srt/mem_cache/l1_transfer/transport/native.cpp"

namespace mock {
struct Stream { CUcontext ctx; std::vector<std::function<void()>> work; size_t done=0; };
struct Event { CUstream stream=nullptr; size_t end=0; bool recorded=false; };
CUcontext context=reinterpret_cast<CUcontext>(0x1234);
std::vector<CUcontext> contexts;
int copies=0, streams=0, events=0, records=0, drains=0, waits=0, destroyed_events=0;
int copy_fail_at=-1, record_fail_at=-1, event_fail_at=-1;
CUstream drain_fail_stream=nullptr;
bool batch_error=false, query_error=false, sync_error=false, drain_error=false;
bool wait_error=false, destroy_event_error=false, destroy_stream_error=false, complete=false;
Stream* stream(CUstream s) { return reinterpret_cast<Stream*>(s); }
Event* event(CUevent e) { return reinterpret_cast<Event*>(e); }
void run(CUstream s, size_t end) {
    auto* q=stream(s);
    while (q->done < end) { auto work=q->work[q->done++]; work(); }
}
CUresult copy(CUdeviceptr dst, CUdeviceptr src, size_t size, CUstream s) {
    if (copies++ == copy_fail_at) return CUDA_ERROR_INVALID_VALUE;
    stream(s)->work.push_back([=] { std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), size); });
    return CUDA_SUCCESS;
}
}

extern "C" {
CUresult CUDAAPI cuGetErrorString(CUresult, const char** out) { *out="injected"; return CUDA_SUCCESS; }
CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* out, CUdevice) { *out=mock::context; return CUDA_SUCCESS; }
CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice) { return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) { mock::contexts.push_back(ctx); return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxPopCurrent(CUcontext* ctx) {
    assert(!mock::contexts.empty()); *ctx=mock::contexts.back(); mock::contexts.pop_back(); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamCreate(CUstream* out, unsigned int) {
    assert(!mock::contexts.empty()); ++mock::streams;
    *out=reinterpret_cast<CUstream>(new mock::Stream{mock::contexts.back(), {}, 0}); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamGetCtx(CUstream s, CUcontext* out) {
    *out=s ? mock::stream(s)->ctx : mock::contexts.back(); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamDestroy(CUstream s) {
    if (mock::destroy_stream_error) return CUDA_ERROR_UNKNOWN;
    auto* q=mock::stream(s); assert(q->done==q->work.size()); delete q; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventCreate(CUevent* out, unsigned int) {
    if (mock::events++ == mock::event_fail_at) return CUDA_ERROR_OUT_OF_MEMORY;
    *out=reinterpret_cast<CUevent>(new mock::Event); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventDestroy(CUevent e) {
    ++mock::destroyed_events;
    if (mock::destroy_event_error) return CUDA_ERROR_UNKNOWN;
    delete mock::event(e); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventRecord(CUevent e, CUstream s) {
    if (mock::records++ == mock::record_fail_at) return CUDA_ERROR_INVALID_VALUE;
    *mock::event(e)={s, mock::stream(s)->work.size(), true}; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWaitEvent(CUstream s, CUevent e, unsigned int) {
    ++mock::waits;
    if (mock::wait_error) return CUDA_ERROR_INVALID_VALUE;
    auto dependency=*mock::event(e); assert(dependency.recorded);
    mock::stream(s)->work.push_back([=] { mock::run(dependency.stream, dependency.end); });
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventQuery(CUevent e) {
    if (mock::query_error) return CUDA_ERROR_UNKNOWN;
    if (!mock::complete) return CUDA_ERROR_NOT_READY;
    auto* event=mock::event(e); assert(event->recorded); mock::run(event->stream,event->end); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventSynchronize(CUevent e) {
    if (mock::sync_error) return CUDA_ERROR_UNKNOWN;
    auto* event=mock::event(e); assert(event->recorded); mock::run(event->stream,event->end); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamSynchronize(CUstream s) {
    ++mock::drains;
    if (mock::drain_error || s==mock::drain_fail_stream) return CUDA_ERROR_UNKNOWN;
    mock::run(s,mock::stream(s)->work.size()); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream s) {
    return mock::copy(dst,src,n,s);
}
#if CUDA_VERSION >= 13000
CUresult CUDAAPI cuMemcpyBatchAsync(CUdeviceptr* dst, CUdeviceptr* src, size_t* sizes, size_t n,
                                  CUmemcpyAttributes* attrs, size_t* indices, size_t na, CUstream s) {
    assert(na==1 && indices[0]==0 && attrs[0].srcAccessOrder==CU_MEMCPY_SRC_ACCESS_ORDER_STREAM);
    for (size_t i=0;i<n;++i) {
        CUresult rc=mock::copy(dst[i],src[i],sizes[i],s);
        if (rc!=CUDA_SUCCESS) return rc;
        if (mock::batch_error) return CUDA_ERROR_INVALID_VALUE; // one copy accepted before failure
    }
    return CUDA_SUCCESS;
}
#endif
}

struct Fixture {
    std::vector<unsigned char> source=std::vector<unsigned char>(64,0x37);
    std::vector<unsigned char> target=std::vector<unsigned char>(64,0xa5);
    std::shared_ptr<ResourcePool> pool=std::make_shared<ResourcePool>(0);
    Region src, dst;
    NvRange ranges[3];
    Fixture() {
        src.pool=dst.pool=pool; src.ctx=dst.ctx=pool->ctx; src.logical=dst.logical=64;
        src.ptr=reinterpret_cast<CUdeviceptr>(source.data()); dst.ptr=reinterpret_cast<CUdeviceptr>(target.data());
        src.imported=true;
        for (int i=0;i<3;++i) ranges[i]={&src,uint64_t(i*16),&dst,uint64_t(i*16),16};
    }
    Batch* submit(CUstream producer=nullptr, bool ready=false) {
        void* out=nullptr;
        assert(nvt_submit(ranges,3,producer,ready,&out)==0 && out);
        return static_cast<Batch*>(out);
    }
    void verify_prefix(size_t length) {
        for(size_t i=0;i<target.size();++i) assert(target[i]==(i<length?source[i]:0xa5));
    }
};

int main(int argc, char** argv) {
    assert(argc==2);
    std::string scenario=argv[1];
    setenv("NVT_RESOURCE_POOL","1",1); setenv("NVT_BATCH_COPY","1",1);
    Fixture f;
    if (scenario=="validation") {
        f.ranges[2].length=65; void* out=nullptr;
        assert(nvt_submit(f.ranges,3,nullptr,0,&out)!=0 && !out);
        assert(mock::copies==0 && mock::streams==0 && f.src.active==0); f.verify_prefix(0);
    } else if (scenario=="wrong-context") {
        auto* q=new mock::Stream{reinterpret_cast<CUcontext>(0x9999),{},0}; void* out=nullptr;
        assert(nvt_submit(f.ranges,3,reinterpret_cast<CUstream>(q),1,&out)!=0 && !out);
        assert(mock::copies==0 && mock::streams==0 && f.src.active==0); delete q;
    } else if (scenario=="success" || scenario=="timeout-late" || scenario=="reuse") {
        Batch* batch=f.submit(); CUstream original=batch->stream; mock::complete=false;
        assert(nvt_wait(batch,0)==0 && batch->state==0 && f.src.active==3);
        assert(nvt_release_batch(batch)!=0); f.verify_prefix(0);
        mock::complete=true; assert(nvt_wait(batch,1)==0 && batch->state==1);
        f.verify_prefix(48); assert(f.src.active==0 && f.dst.active==0);
        assert(nvt_release_batch(batch)==0 && f.pool->idle.size()==1);
        if (scenario=="reuse") {
            batch=f.submit(); assert(batch->stream==original && mock::streams==1 && mock::events==1);
            assert(nvt_wait(batch,-1)==0 && batch->state==1); assert(nvt_release_batch(batch)==0);
        }
    } else if (scenario=="producer-order") {
        CUstream producer; { Context ctx(f.pool->ctx); assert(cuStreamCreate(&producer,0)==CUDA_SUCCESS); }
        mock::stream(producer)->work.push_back([&] { std::fill(f.source.begin(),f.source.end(),0x42); });
        f.src.imported=false; f.dst.imported=true;
        Batch* batch=f.submit(producer,true);
        assert(mock::waits==1 && mock::drains==0 && f.source[0]==0x37);
        mock::complete=true; assert(nvt_wait(batch,1)==0 && batch->state==1);
        f.verify_prefix(48); assert(f.target[0]==0x42);
        assert(nvt_release_batch(batch)==0 && f.pool->idle[0].ready);
        assert(cuStreamDestroy(producer)==CUDA_SUCCESS);
    } else if (scenario=="release-event" || scenario=="release-stream") {
        setenv("NVT_RESOURCE_POOL","0",1);
        Batch* batch=f.submit(); mock::complete=true; assert(nvt_wait(batch,1)==0);
        mock::destroy_event_error=scenario=="release-event";
        mock::destroy_stream_error=scenario=="release-stream";
        assert(nvt_release_batch(batch)!=0 && batch->state==3 && f.src.poisoned && f.dst.poisoned);
        if (scenario=="release-stream") assert(batch->event==nullptr);
        int destroys=mock::destroyed_events;
        assert(nvt_release_batch(batch)!=0 && mock::destroyed_events==destroys && f.pool->idle.empty());
    } else {
        const bool uncertain=scenario.find("uncertain")!=std::string::npos;
        mock::drain_error=uncertain;
        CUstream producer=nullptr;
        bool use_producer=scenario.find("producer-")==0;
        if (use_producer) {
            Context ctx(f.pool->ctx); assert(cuStreamCreate(&producer,0)==CUDA_SUCCESS);
            mock::stream(producer)->work.push_back([&] { std::fill(f.source.begin(),f.source.end(),0x42); });
            if (scenario=="producer-wait-drain-uncertain") {
                mock::drain_error=false; mock::drain_fail_stream=producer;
            }
        }
        if (scenario.find("partial-seq")==0) { setenv("NVT_BATCH_COPY","0",1); mock::copy_fail_at=1; }
        else if (scenario.find("partial-batch")==0) mock::batch_error=true;
        else if (scenario.find("record-")==0) mock::record_fail_at=0;
        else if (scenario.find("producer-record")==0) mock::record_fail_at=0;
        else if (scenario.find("producer-create")==0) mock::event_fail_at=1;
        else if (scenario.find("producer-wait")==0) mock::wait_error=true;
        else assert(scenario.find("query-")==0 || scenario.find("wait-")==0);
        Batch* batch=f.submit(producer,use_producer);
        if (scenario.find("query-")==0) {
            mock::query_error=true; int state=0; assert(nvt_poll(batch,&state)==0);
        }
        if (scenario.find("wait-")==0) {
            mock::sync_error=true; assert(nvt_wait(batch,-1)==0);
        }
        assert(mock::drains==(use_producer?2:1) && batch->state==(uncertain?3:2));
        assert(batch->failure.find("stream drain result=")!=std::string::npos);
        assert(f.src.poisoned==uncertain && f.dst.poisoned==uncertain);
        assert(f.src.active==(uncertain?3u:0u));
        if (uncertain) {
            assert(nvt_release_batch(batch)!=0 && f.pool->idle.empty()); f.verify_prefix(0);
        } else {
            if (use_producer) assert(f.source[0]==0x42);
            size_t copied=use_producer?0:(scenario.find("partial-")==0?16:48);
            f.verify_prefix(copied);
            assert(nvt_release_batch(batch)==0 && f.pool->idle.empty());
        }
        if (producer && !uncertain) assert(cuStreamDestroy(producer)==CUDA_SUCCESS);
    }
    std::cout << "PASS " << scenario << std::endl;
}
