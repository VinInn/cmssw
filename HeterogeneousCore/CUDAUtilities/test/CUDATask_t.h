#include "HeterogeneousCore/CUDAUtilities/interface/CUDATask.h"
#include <cstdio>

using namespace cms::cuda;


__global__ void one(int32_t *d_in, int32_t *d_out, int32_t n) {
  auto init = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = iWork * blockDim.x + threadIdx.x;
    for (int i = first; i < n; i += gridDim.x * blockDim.x) {
      d_in[i] = -1;
      d_out[i] = -5;
    }
    d_in[3333] = -4;  // touch it everywhere
    if (15 == d_in[1234])
      d_in[1234] = 33;
    if (15 == d_out[200234])
      d_out[200234] = 33;
  };

  init(blockIdx.x);
}

__global__ void two(int32_t *d_in, int32_t *d_out, int32_t *one, int32_t *blocks, int32_t n) {
  auto setIt = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = iWork * blockDim.x + threadIdx.x;
    if (0 == threadIdx.x)
      blocks[blockIdx.x] = 1;
    for (int i = first; i < n; i += gridDim.x * blockDim.x) {
      d_in[i] = 5;
      ++one[i];
    }
    d_in[5324] = 4;  // should fail
    if (15 == d_in[10234])
      d_in[10234] = 33;
    if (15 == d_out[10234])
      d_out[10234] = 33;
  };

  setIt(blockIdx.x);
}

__global__ void three(int32_t *d_in, int32_t *d_out, int32_t n) {
  auto testIt1 = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = (gridDim.x - iWork - 1) * blockDim.x + threadIdx.x;
    for (int i = first; i < n; i += gridDim.x * blockDim.x)
      if (5 == d_in[i])
        d_out[i] = 5;
  };

  testIt1(blockIdx.x);
}

template <int N>
__global__ void testTask(int32_t *d_in, int32_t *d_out, int32_t *one, int32_t *blocks, int32_t n, CUDATask *task) {
  auto voidTail = []() {};

  auto init = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = iWork * blockDim.x + threadIdx.x;
    for (int i = first; i < n; i += gridDim.x * blockDim.x) {
      d_in[i] = -1;
      d_out[i] = -5;
    }
    d_in[3333] = -4;  // touch it everywhere
    if (15 == d_in[1234])
      d_in[1234] = 33;
    if (15 == d_out[200234])
      d_out[200234] = 33;
  };

  auto setIt = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = iWork * blockDim.x + threadIdx.x;
    if (0 == threadIdx.x)
      blocks[blockIdx.x] = 1;
    for (int i = first; i < n; i += gridDim.x * blockDim.x) {
      d_in[i] = 5;
      ++one[i];
    }
    d_in[5324] = 4;  // should fail
    if (15 == d_in[10234])
      d_in[10234] = 33;
    if (15 == d_out[10234])
      d_out[10234] = 33;
  };

  auto testIt1 = [&](int32_t iWork) {
    // standard loop  (iWork instead of blockIdx.x)
    auto first = (gridDim.x - iWork - 1) * blockDim.x + threadIdx.x;
    for (int i = first; i < n; i += gridDim.x * blockDim.x)
      if (5 == d_in[i])
        d_out[i] = 5;
  };

  task[0].doit(init, voidTail);
  task[1].doit(setIt, voidTail);
  task[2].doit(testIt1, voidTail);
}


#include <cooperative_groups.h>

template<typename KERNEL>
__forceinline__
std::pair<int,int> coopKernelConfig(KERNEL kernel, int dev=0) {
   assert(dev<16);
   constexpr int nThreads = 256;
   static cudaDeviceProp deviceProp[16];
   static int numBlocksPerSm[16] = {0};
   if (0==numBlocksPerSm[dev]) { // not fully thread safe....)
     cudaGetDeviceProperties(&deviceProp[dev], 0);
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm[dev], kernel, nThreads, 0);
   }
   return std::make_pair(deviceProp[dev].multiProcessorCount*numBlocksPerSm[dev],nThreads);
 
}

template <int N>
__global__ void testCoop(int32_t *d_in, int32_t *d_out, int32_t *one, int32_t *blocks, int32_t n) {

  using namespace cooperative_groups;

  auto init = [&]() {
    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = first; i < n; i += gridDim.x * blockDim.x) {
      d_in[i] = -1;
      d_out[i] = -5;
    }
    d_in[3333] = -4;  // touch it everywhere
    if (15 == d_in[1234])
      d_in[1234] = 33;
    if (15 == d_out[200234])
      d_out[200234] = 33;
  };

  auto setIt = [&]() {
    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 == threadIdx.x)
      blocks[blockIdx.x] = 1;
    for (int i = first; i < n; i += gridDim.x * blockDim.x) {
      d_in[i] = 5;
      ++one[i];
    }
    d_in[5324] = 4;  // should fail
    if (15 == d_in[10234])
      d_in[10234] = 33;
    if (15 == d_out[10234])
      d_out[10234] = 33;
  };

  auto testIt1 = [&]() {
    auto first = (gridDim.x - blockIdx.x - 1) * blockDim.x + threadIdx.x;
    for (int i = first; i < n; i += gridDim.x * blockDim.x)
      if (5 == d_in[i])
        d_out[i] = 5;
  };

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  init();
  grid.sync();
  setIt();
  grid.sync();
  testIt1();

}



__global__ void verify(int32_t *d_out, int32_t *one, int32_t n) {
  auto first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = first; i < n; i += gridDim.x * blockDim.x) {
    if (5 != d_out[i])
      printf("out failed %d %d/%d\n", i, blockIdx.x, threadIdx.x);
    if (1 != one[i])
      printf("one failed %d %d/%d\n", i, blockIdx.x, threadIdx.x);
  }
}
