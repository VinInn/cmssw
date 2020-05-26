#ifndef HeterogeneousCore_CUDAUtilities_interface_CUDATask_h
#define HeterogeneousCore_CUDAUtilities_interface_CUDATask_h

#include <cstdint>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include <cooperative_groups.h>

namespace cms {
  namespace cuda {
struct CoopKernelConfig {
  explicit CoopKernelConfig(int nthreads) : nThreads(nthreads) {}

  template <typename KERNEL>
  inline std::pair<int, int> getConfig(KERNEL kernel, int dev=0) {
    assert(dev < 16);
    if (0 == numBlocksPerSm[dev]) {
    //  std::cout << " checking device " << std::endl;
      cudaGetDeviceProperties(&deviceProp[dev], 0);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm[dev], kernel, nThreads, 0);
    }
    return std::make_pair(deviceProp[dev].multiProcessorCount * numBlocksPerSm[dev], nThreads);
  }

  const int nThreads = 256;
  cudaDeviceProp deviceProp[16];
  int numBlocksPerSm[16] = {0};
};

}
}

namespace cms {
  namespace cuda {

    class CUDATask {
    public:
      // better to be called in the tail of the previous task...
      __device__ void __forceinline__ zero() {
        nWork = 0;
        nDone = 0;
        allDone = 0;
      }

      template <typename BODY, typename TAIL>
      __device__ void __forceinline__ doit(BODY body, TAIL tail) {
        __shared__ int iWork;
        bool done = false;
        __shared__ bool isLastBlockDone;

        isLastBlockDone = false;

        /*     
     //  fast jump for late blocks??  (worth only if way to many blocks scheduled)
     if (0 == threadIdx.x) {
          iWork = nWork;
     }
     __syncthreads();
     done = iWork >=int(gridDim.x);
     */

        while (__syncthreads_and(!done)) {
          if (0 == threadIdx.x) {
            iWork = atomicAdd(&nWork, 1);
          }
          __syncthreads();

          assert(iWork >= 0);

          done = iWork >= int(gridDim.x);

          if (!done) {
            body(iWork);

            __threadfence();

            // count blocks that finished
            if (0 == threadIdx.x) {
              auto value = atomicAdd(&nDone, 1);  // block counter
              isLastBlockDone = (value == (int(gridDim.x) - 1));
            }

          }  // done
        }    // while

        if (isLastBlockDone) {
          assert(0 == (allDone));

          assert(int(gridDim.x) == nDone);

          // good each block has done its work and now we are left in last block
          tail();
          __syncthreads();
          if (0 == threadIdx.x)
            allDone = 1;
          __syncthreads();
        }

        // we need to wait the one above...
        while (0 == (allDone)) {
          __threadfence();
        }

        __syncthreads();  // at some point we must decide who sync

        assert(1 == allDone);
      }

    public:
      int32_t nWork;
      int32_t nDone;
      volatile int32_t allDone;  // can be bool
    };

  }  // namespace cuda
}  // namespace cms

#endif
