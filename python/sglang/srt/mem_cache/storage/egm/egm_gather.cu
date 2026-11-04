// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to SGLang project

// One-launch gather copy for many small ranges; SM loads read EGM/Fabric mappings directly.
#include <cuda_runtime.h>
#include <cstdint>

// src / dst / size: device arrays of count device pointers and byte counts.
__global__ void egm_gather_kernel(const uint64_t *src, const uint64_t *dst, const uint64_t *size, uint64_t count) {
    for (uint64_t i = blockIdx.x; i < count; i += gridDim.x) {
        const char *s = reinterpret_cast<const char *>(src[i]);
        char *d = reinterpret_cast<char *>(dst[i]);
        uint64_t n = size[i];
        if ((reinterpret_cast<uintptr_t>(s) | reinterpret_cast<uintptr_t>(d) | n) & 15) {
            for (uint64_t j = threadIdx.x; j < n; j += blockDim.x) d[j] = s[j];
        } else {
            const uint4 *s4 = reinterpret_cast<const uint4 *>(s);
            uint4 *d4 = reinterpret_cast<uint4 *>(d);
            for (uint64_t j = threadIdx.x; j < n / 16; j += blockDim.x) d4[j] = s4[j];
        }
    }
}

extern "C" int egm_gather_launch(const uint64_t *src, const uint64_t *dst, const uint64_t *size, uint64_t count,
                                 void *stream) {
    unsigned blocks = count < 65535 ? static_cast<unsigned>(count) : 65535u;
    egm_gather_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(src, dst, size, count);
    return static_cast<int>(cudaGetLastError());
}
