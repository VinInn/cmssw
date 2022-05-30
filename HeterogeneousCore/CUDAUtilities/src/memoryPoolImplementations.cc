// This file has to be split in N, one for each backend

#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/Queue.h"

#include<new>
#include<memory>

// POSIX implementations

#include <cstring>

namespace {

  std::unique_ptr<SimplePoolAllocatorImpl<PosixAlloc>> cpuPool;

}

namespace memoryPool {

    void initCPU() {
      constexpr int poolSize = 128 * 1024;
      cpuPool = std::make_unique<SimplePoolAllocatorImpl<PosixAlloc>>(poolSize);
    }

    void shutdownCPU() {
      cpuPool.reset();
    }


   Queue getCPUQueue() {
     OpaqueQueue q{cpuPool.get(), cpuPool.get(), nullptr, DeviceAPI::posix};
     return std::make_shared<OpaqueQueue>(q);
   }

   void cpuCopy(void * dst,  void const * src,  uint64_t size) {
      ::memcpy(dst,src,size);
   }
 
}

//----------------------------------------------------------------------------------------------------------------------

// CUDA implementations

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"

#include <iostream>

namespace {

  //  free callback
  void CUDART_CB freeCallback(cudaStream_t streamId, cudaError_t status, void *p) {
    //void CUDART_CB freeCallback(void *p) {  // alternative signature in case of use of cudaLaunchHostFunc
    if (status != cudaSuccess) {
      std::cout << "Error in free callaback in stream " << streamId << std::endl;
      auto error = cudaGetErrorName(status);
      auto message = cudaGetErrorString(status);
      std::cout << " error " << error << ": " << message << std::endl;
    }
    // std::cout << "free callaback for stream " << streamId << std::endl;
    auto payload = (memoryPool::Payload *)(p);
    poolDetails::freeAsync(payload);
  }

}  // namespace

struct CudaAlloc {
  static void scheduleFree(memoryPool::Payload *payload, void *stream) {
    // std::cout    << "schedule free for stream " <<  stream <<std::endl;
    if (!stream)
      std::cout << "???? schedule free for stream " << stream << std::endl;
    if (cudaStreamQuery((cudaStream_t)(stream)) == cudaSuccess)
      poolDetails::freeAsync(payload);
    else
      cudaCheck(cudaStreamAddCallback((cudaStream_t)(stream), freeCallback, payload, 0));
    // cudaCheck(cudaLaunchHostFunc(stream, freeCallback, payload)); // alternative
  }
};

struct CudaDeviceAlloc : public CudaAlloc {
  using Pointer = void *;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMalloc(&p, size);
    // std::cout << "alloc " << size << ((err == cudaSuccess) ? " ok" : " err") << std::endl;
    return err == cudaSuccess ? p : nullptr;
  }
  static void free(Pointer ptr) {
    if (ptr == nullptr)
      std::cout << "free nullptr???" << std::endl;
    auto err = cudaFree(ptr);
    // std::cout << "free" << ((err == cudaSuccess) ? " ok" : " err") <<std::endl;
    if (err != cudaSuccess)
      std::cout << " error in cudaFree??" << std::endl;
  }
};

struct CudaHostAlloc : public CudaAlloc {
  using Pointer = void *;

  static Pointer alloc(size_t size) {
    Pointer p = nullptr;
    auto err = cudaMallocHost(&p, size);
    // std::cout << "alloc H " << size << ((err == cudaSuccess) ? " ok" : " err") << std::endl;
    return err == cudaSuccess ? p : nullptr;
  }
  static void free(Pointer ptr) { cudaFreeHost(ptr); }
};


namespace {

  std::unique_ptr<SimplePoolAllocatorImpl<CudaHostAlloc>> hostPool;

  using DevicePool = SimplePoolAllocatorImpl<CudaDeviceAlloc>;
  std::vector<std::unique_ptr<DevicePool>> devicePools;

  void initDevicePools(int size) {
    int devices = 0;
    auto status = cudaGetDeviceCount(&devices);
    if (status == cudaSuccess && devices > 0) {
      devicePools.reserve(devices);
      for (int i = 0; i < devices; ++i)
        devicePools.emplace_back(new DevicePool(size));
    }
  }

  DevicePool *getDevicePool() {
    int dev = -1;
    cudaGetDevice(&dev);
    return devicePools[dev].get();
  }


  using CudaQueue = memoryPool::OpaqueQueue; // no derived class yet....

}  // namespace


namespace memoryPool {

    void initCUDA() {
      constexpr int poolSize = 128 * 1024;
      initDevicePools(poolSize);
      hostPool = std::make_unique<SimplePoolAllocatorImpl<CudaHostAlloc>>(poolSize);
    }

    void shutdownCUDA() {
      devicePools.clear();
      hostPool.reset();
    }

    void dumpStatCUDA() {
      std::cout << "device pool" << std::endl;
      getDevicePool()->dumpStat();
      std::cout << "host pool" << std::endl;
      hostPool->dumpStat();
    }


 Queue getCudaQueue(void * ostream) {
    // auto stream =  cudaStream_t(ostream);  // eventually
    return std::make_shared<CudaQueue>(CudaQueue{getDevicePool(), hostPool.get() , ostream, DeviceAPI::cuda});

 }


 void cudaCopy(void *    dst,  void const * src,     uint64_t size, void * stream) {
    cudaCheck(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, cudaStream_t(stream)));
 }


    void copyToHost(void * dst,  void const * src,  uint64_t size, Queue queue) {
        cudaCheck(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, cudaStream_t(queue->m_stream)));
    }

    void copyToDevice(void * dst,  void const * src,  uint64_t size, Queue queue) {
      cudaCheck(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, cudaStream_t(queue->m_stream)));
    }



}


//----------------------------------------------------------------------------------------------------------------------


// HIP implementations 
// just take cuda and sed-it
namespace memoryPool {

  Queue getHipQueue(void *) {
     return Queue();
  }


 void hipCopy(void *    dst,  void const * src,     uint64_t size, void * stream) {}

}
