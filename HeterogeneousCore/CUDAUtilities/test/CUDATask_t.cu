#include "CUDATask_t.h"

#include <iostream>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include <chrono>

using namespace std::chrono;


int main() {

  cms::cudatest::requireDevices();

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::cout << "Running on Device " << ' ' << " with "
  << "\nmultiProcessorCount " << deviceProp.multiProcessorCount
  << "\nmaxThreadsPerMultiProcessor " << deviceProp.maxThreadsPerMultiProcessor 
  << std::endl;

  int32_t *d_in;
  int32_t *d_out1;
  int32_t *d_out2;

  int32_t num_items = 1000*1000;

  cudaCheck(cudaMalloc(&d_in, num_items * sizeof(uint32_t)));
  cudaCheck(cudaMalloc(&d_out1, num_items * sizeof(uint32_t)));
  cudaCheck(cudaMalloc(&d_out2, num_items * sizeof(uint32_t)));

  auto nthreads = 256;
  auto nblocks = (num_items + nthreads - 1) / nthreads;

  int32_t * blocks;
  int32_t * h_blocks;
  cudaCheck(cudaMalloc(&blocks, nblocks * sizeof(uint32_t)));
  cudaCheck(cudaMemset(blocks, 0, nblocks * sizeof(uint32_t)));
  cudaCheck(cudaMallocHost(&h_blocks, nblocks * sizeof(uint32_t)));

  CUDATask * task;  // 3 of them
  cudaCheck(cudaMalloc(&task, 3*sizeof(CUDATask)));
  cudaCheck(cudaMemset(task, 0, 3*sizeof(CUDATask)));

  cudaCheck(cudaMemset(d_in, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out1, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out2, 0, num_items*sizeof(int32_t)));

  {
  std::cout << "scheduling " << nblocks << " blocks of " << nthreads << " threads"<< std::endl;
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  one<<<nblocks, nthreads, 0>>>(d_in, d_out1, num_items);
  two<<<nblocks, nthreads, 0>>>(d_in, d_out1, d_out2, blocks, num_items);
  three<<<nblocks, nthreads, 0>>>(d_in, d_out1, num_items);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  cudaCheck(cudaGetLastError());
  verify<<<nblocks, nthreads, 0>>>(d_out1, d_out2, num_items);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(h_blocks, blocks, nblocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  int s=0; for (int i=0; i<nblocks;++i) s += h_blocks[i];
  std::cout << "standard kernel used " << s    << " blocks" << std::endl;
  auto delta = duration_cast<duration<double>>(t2 - t1).count();
  std::cout << "three kernels took " << delta << std::endl;
  }


  {
  cudaCheck(cudaMemset(d_in, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out1, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out2, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(task, 0, 3*sizeof(CUDATask)));
  cudaCheck(cudaMemset(blocks, 0, nblocks * sizeof(uint32_t)));

  std::cout << "scheduling " << nblocks << " blocks of " << nthreads << " threads"<< std::endl;
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  testTask<1> <<<nblocks, nthreads, 0>>>(d_in, d_out1, d_out2, blocks, num_items, task);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  cudaCheck(cudaGetLastError());
  verify<<<nblocks, nthreads, 0>>>(d_out1, d_out2, num_items);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize(); 
  cudaMemcpy(h_blocks, blocks, nblocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  int s=0; for (int i=0; i<nblocks;++i) s += h_blocks[i];
  std::cout << "task kernel used " << s    << " blocks" << std::endl;
  auto delta = duration_cast<duration<double>>(t2 - t1).count();
  std::cout << "task kernel took " << delta << std::endl;
  }

  {
     nblocks /= 32;
  cudaCheck(cudaMemset(d_in, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out1, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(d_out2, 0, num_items*sizeof(int32_t)));
  cudaCheck(cudaMemset(task, 0, 3*sizeof(CUDATask)));
  cudaCheck(cudaMemset(blocks, 0, nblocks * sizeof(uint32_t)));

  std::cout << "scheduling " << nblocks << " blocks of " << nthreads << " threads"<< std::endl;
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  testTask<2> <<<nblocks, nthreads, 0>>>(d_in, d_out1, d_out2, blocks, num_items, task);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  cudaCheck(cudaGetLastError());
  verify<<<nblocks, nthreads, 0>>>(d_out1, d_out2, num_items);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(h_blocks, blocks, nblocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  int s=0; for (int i=0; i<nblocks;++i) s += h_blocks[i];
  std::cout << "task kernel used " << s << " blocks" << std::endl;
  auto delta = duration_cast<duration<double>>(t2 - t1).count();
  std::cout << "task kernel took " << delta << std::endl;
  }

  return 0;
};
 
