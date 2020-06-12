#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <cstdint>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "HeterogeneousCore/CUDAUtilities/interface/CUDATask.h"
#include <cooperative_groups.h>

#ifdef __CUDA_ARCH__

template <typename T>
__device__ void __forceinline__ warpPrefixScan(T const* __restrict__ ci, T* __restrict__ co, uint32_t i, uint32_t mask) {
  // ci and co may be the same
  auto x = ci[i];
  auto laneId = threadIdx.x & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
}

template <typename T>
__device__ void __forceinline__ warpPrefixScan(T* c, uint32_t i, uint32_t mask) {
  auto x = c[i];
  auto laneId = threadIdx.x & 0x1f;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

#endif

namespace cms {
  namespace cuda {

    // limited to 32*32 elements....
    template <typename T>
    __host__ __device__ __forceinline__ void blockPrefixScan(T const* __restrict__ ci,
                                                             T* __restrict__ co,
                                                             uint32_t size,
                                                             T* ws
#ifndef __CUDA_ARCH__
                                                             = nullptr
#endif
    ) {
#ifdef __CUDA_ARCH__
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDim.x % 32);
      auto first = threadIdx.x;
      auto mask = __ballot_sync(0xffffffff, first < size);

      for (auto i = first; i < size; i += blockDim.x) {
        warpPrefixScan(ci, co, i, mask);
        auto laneId = threadIdx.x & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = co[i];
        mask = __ballot_sync(mask, i + blockDim.x < size);
      }
      __syncthreads();
      if (size <= 32)
        return;
      if (threadIdx.x < 32)
        warpPrefixScan(ws, threadIdx.x, 0xffffffff);
      __syncthreads();
      for (auto i = first + 32; i < size; i += blockDim.x) {
        auto warpId = i / 32;
        co[i] += ws[warpId - 1];
      }
      __syncthreads();
#else
      co[0] = ci[0];
      for (uint32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
#endif
    }

    // same as above, may remove
    // limited to 32*32 elements....
    template <typename T>
    __host__ __device__ __forceinline__ void blockPrefixScan(T* c,
                                                             uint32_t size,
                                                             T* ws
#ifndef __CUDA_ARCH__
                                                             = nullptr
#endif
    ) {
#ifdef __CUDA_ARCH__
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDim.x % 32);
      auto first = threadIdx.x;
      auto mask = __ballot_sync(0xffffffff, first < size);

      for (auto i = first; i < size; i += blockDim.x) {
        warpPrefixScan(c, i, mask);
        auto laneId = threadIdx.x & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = c[i];
        mask = __ballot_sync(mask, i + blockDim.x < size);
      }
      __syncthreads();
      if (size <= 32)
        return;
      if (threadIdx.x < 32)
        warpPrefixScan(ws, threadIdx.x, 0xffffffff);
      __syncthreads();
      for (auto i = first + 32; i < size; i += blockDim.x) {
        auto warpId = i / 32;
        c[i] += ws[warpId - 1];
      }
      __syncthreads();
#else
      for (uint32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
#endif
    }

#ifdef __CUDA_ARCH__
    // see https://stackoverflow.com/questions/40021086/can-i-obtain-the-amount-of-allocated-dynamic-shared-memory-from-within-a-kernel/40021087#40021087
    __device__ __forceinline__ unsigned dynamic_smem_size() {
      unsigned ret;
      asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
      return ret;
    }
#endif

    // in principle not limited....
    template <typename T>
    __global__ void multiBlockPrefixScan(T const* ci, T* co, int32_t size, int32_t* pc) {
      __shared__ T ws[32];
#ifdef __CUDA_ARCH__
      assert(sizeof(T) * (size / blockDim.x) <= dynamic_smem_size());  // size of psum below
#endif
      assert(blockDim.x * gridDim.x >= size);
      // first each block does a scan
      int off = blockDim.x * blockIdx.x;
      if (size - off > 0)
        blockPrefixScan(ci + off, co + off, std::min(int(blockDim.x), size - off), ws);

      // count blocks that finished
      __shared__ bool isLastBlockDone;
      if (0 == threadIdx.x) {
        auto value = atomicAdd(pc, 1);  // block counter
        isLastBlockDone = (value == (int(gridDim.x) - 1));
      }

      __syncthreads();

      if (!isLastBlockDone)
        return;

      assert(int(gridDim.x) == *pc);

      // good each block has done its work and now we are left in last block

      // let's get the partial sums from each block
      extern __shared__ T psum[];
      int nChunks = size / blockDim.x;
      for (int i = threadIdx.x; i < nChunks; i += blockDim.x) {
        auto j = blockDim.x * i + blockDim.x - 1;
        assert(j < size);
        psum[i] = co[j];
      }
      __syncthreads();
      blockPrefixScan(psum, psum, nChunks, ws);

      // now it would have been handy to have the other blocks around...
      for (int i = threadIdx.x + blockDim.x, k = 0; i < size; i += blockDim.x, ++k) {
        assert(k < nChunks);
        co[i] += psum[k];
      }
    }

    // in principle not limited....
    template <typename T>
    __device__ void __forceinline__ multiTaskPrefixScan(T const* gci, T* gco, int32_t size, CUDATask& task, T* gpsum) {
      volatile auto ci = gci;
      volatile auto co = gco;
      volatile auto psum = gpsum;

      __shared__ T ws[32];

      auto body = [&](int32_t iWork) {
        // first each block does a scan
        for (int off = blockDim.x * iWork; off < size; off += blockDim.x * gridDim.x) {
          blockPrefixScan(ci + off, co + off, std::min(int(blockDim.x), size - off), ws);
        }
      };

      auto tail = [&]() {
        // let's get the partial sums from each block
        int nChunks = size / blockDim.x;
        for (int i = threadIdx.x; i < nChunks; i += blockDim.x) {
          auto j = blockDim.x * i + blockDim.x - 1;
          assert(j < size);
          psum[i] = co[j];
        }
        __syncthreads();
        blockPrefixScan(psum, psum, nChunks, ws);
      };

      task.doit(body, tail);

      // now it is very handy to have the other blocks around...
      auto first = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
      for (int i = first; i < size; i += gridDim.x * blockDim.x) {
        assert(blockIdx.x < size / blockDim.x);
        co[i] += psum[blockIdx.x];
      }
    }

    // in principle not limited....
    template <typename T>
    __global__ void multiTaskPrefixScanKernel(T const* ci, T* co, int32_t size, CUDATask* task, T* psum) {
      multiTaskPrefixScan(ci, co, size, *task, psum);
    }

    // in principle not limited....
    template <typename T>
    __device__ void __forceinline__ coopPrefixScan(T const* gci, T* gco, int32_t size, T* gpsum) {
      using namespace cooperative_groups;

      volatile auto ci = gci;
      volatile auto co = gco;
      volatile auto psum = gpsum;

      __shared__ T ws[32];

      auto body = [&]() {
        // first each block does a scan
        for (int off = blockDim.x * blockIdx.x; off < size; off += blockDim.x * gridDim.x) {
          blockPrefixScan(ci + off, co + off, std::min(int(blockDim.x), size - off), ws);
        }
      };

      auto tail = [&]() {
        // let's get the partial sums from each block
        int nChunks = size / blockDim.x;
        for (int i = threadIdx.x; i < nChunks; i += blockDim.x) {
          auto j = blockDim.x * i + blockDim.x - 1;
          assert(j < size);
          psum[i] = co[j];
        }
        __syncthreads();
        blockPrefixScan(psum, psum, nChunks, ws);
      };

      cooperative_groups::grid_group grid = cooperative_groups::this_grid();
      body();
      grid.sync();
      if (0 == blockIdx.x)
        tail();
      grid.sync();

      // now it is very handy to have the other blocks around...
      auto first = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
      for (int i = first; i < size; i += gridDim.x * blockDim.x) {
        assert(blockIdx.x < size / blockDim.x);
        co[i] += psum[blockIdx.x];
      }
    }

    // in principle not limited....
    template <typename T>
    __global__ void coopPrefixScanKernel(T const* ci, T* co, int32_t size, T* psum) {
      coopPrefixScan(ci, co, size, psum);
    }

  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
