#include <Aten/Aten.h>
#include <Aten/DeviceGaurd.h>
#include <Aten/Dispatch.h>
#include <Aten/cuda/CudaContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include "c10/core/ScalarType.h"
#include "c10/util/BFloat16.h"

#ifndef USE_ROCM
#include <mma.h>
#endif
#include <cub/cub.cuh>

#include "fbgemm_gpu/utils/cuda_block_count.h"
#include "fbgemm_gpu/utils/vec_quant.cuh"

#include <torch/torch.h>

template <typename func_t>
void set_gpu_max_dynamic_shared_memory(
    func_t kernel,
    const int smen_bytes,
    const int device) {
    // V100: 96 KB; A100: 160 KB; H100: 228 KB.
    int max_shared_bytes = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_shared_bytes,
    #ifndef __HIP_PLATFORM_AMD__
           cudaDevAttrMaxSharedMemoryPerBlockOptin,
    #else
           hipDeviceAttributeMaxSharedMemoryPerBlock,
    #endif
          device));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    TORCH_CHECK(
        smen_bytes <= max_shared_bytes,
        "Try to allocate", 
        smen_bytes / 1024,
        " KB of shared memory but only",
        max_shared_bytes / 1024,
        " KB is available");

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        (void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smen_bytes));
    }

    namespace fbgemm_gpu {
        template <int KVQuantNumGroups = 1>
        __global__ void dequantize_int4_cache_kernel(
            at::PackedTensorAccessor64<unit8_t, 4, at::RestrictPtrTraits>
                cache_K, // [B] [MAX_T][N_KVH][D_H // G]
                // 65
        )
    }