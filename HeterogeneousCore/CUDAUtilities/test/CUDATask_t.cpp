#include "CUDATask_t.h"

#include <iostream>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include <chrono>

using namespace std::chrono;


int main() {

  std::cout << "Running on CPU "
  << std::endl;

  int32_t num_items = 1000*1000;

  int32_t *d_in = (int32_t *)malloc(num_items * sizeof(uint32_t));
  int32_t *d_out1 = (int32_t *)malloc(num_items * sizeof(uint32_t));
  int32_t *d_out2 = (int32_t *)malloc(num_items * sizeof(uint32_t));

  auto nthreads = 256;
  auto nblocks = (num_items + nthreads - 1) / nthreads;

  int32_t * blocks = (int32_t *)malloc(nblocks * sizeof(uint32_t));
  memset(blocks, 0, nblocks * sizeof(uint32_t));
  int32_t * h_blocks = blocks;

  CUDATask * task = (CUDATask *)malloc(3*sizeof(CUDATask));  // 3 of them
  memset(task, 0, 3*sizeof(CUDATask));

  memset(d_in, 0, num_items*sizeof(int32_t));
  memset(d_out1, 0, num_items*sizeof(int32_t));
  memset(d_out2, 0, num_items*sizeof(int32_t));

  {
  std::cout << "scheduling " << nblocks << " blocks of " << nthreads << " threads"<< std::endl;
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  one(d_in, d_out1, num_items);
  two(d_in, d_out1, d_out2, blocks, num_items);
  three(d_in, d_out1, num_items);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  verify(d_out1, d_out2, num_items);
  int s=0; for (int i=0; i<nblocks;++i) s += h_blocks[i];
  std::cout << "standard kernel used " << s    << " blocks" << std::endl;
  auto delta = duration_cast<duration<double>>(t2 - t1).count();
  std::cout << "three kernels took " << delta << std::endl;
  }


  {
  memset(d_in, 0, num_items*sizeof(int32_t));
  memset(d_out1, 0, num_items*sizeof(int32_t));
  memset(d_out2, 0, num_items*sizeof(int32_t));
  memset(task, 0, 3*sizeof(CUDATask));
  memset(blocks, 0, nblocks * sizeof(uint32_t));

  std::cout << "scheduling " << nblocks << " blocks of " << nthreads << " threads"<< std::endl;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  testTask<1>(d_in, d_out1, d_out2, blocks, num_items, task);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  verify(d_out1, d_out2, num_items);
  int s=0; for (int i=0; i<nblocks;++i) s += h_blocks[i];
  std::cout << "task kernel used " << s    << " blocks" << std::endl;
  auto delta = duration_cast<duration<double>>(t2 - t1).count();
  std::cout << "task kernel took " << delta << std::endl;
  }

  {
     nblocks /= 32;
  memset(d_in, 0, num_items*sizeof(int32_t));
  memset(d_out1, 0, num_items*sizeof(int32_t));
  memset(d_out2, 0, num_items*sizeof(int32_t));
  memset(task, 0, 3*sizeof(CUDATask));
  memset(blocks, 0, nblocks * sizeof(uint32_t));

  std::cout << "scheduling " << nblocks << " blocks of " << nthreads << " threads"<< std::endl;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  testTask<2>(d_in, d_out1, d_out2, blocks, num_items, task);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  verify(d_out1, d_out2, num_items);
  int s=0; for (int i=0; i<nblocks;++i) s += h_blocks[i];
  std::cout << "task kernel used " << s    << " blocks" << std::endl;
  auto delta = duration_cast<duration<double>>(t2 - t1).count();
  std::cout << "task kernel took " << delta << std::endl;
  }

  return 0;
};
 
