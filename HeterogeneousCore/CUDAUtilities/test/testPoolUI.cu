#include "HeterogeneousCore/CUDAUtilities/interface/memoryPool.h"

#include <iostream>

int main() {
  {
    int devices = 0;
    auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || 0 == devices)
      return 0;
    std::cout << "found " << devices << " cuda devices" << std::endl;
  }
  const int NUMTHREADS = 1;

  printf("Using CUDA %d\n", CUDART_VERSION);
  int cuda_device = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, cuda_device);
  printf("CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);

  cudaStream_t streams[NUMTHREADS];

  for (int i = 0; i < NUMTHREADS; i++) {
    cudaStreamCreate(&(streams[i]));
  }

  memoryPool::initCUDA();
  memoryPool::dumpStatCUDA();
  memoryPool::initCPU();

  auto& stream = streams[0];
  auto queue = memoryPool::getQueue(stream, memoryPool::DeviceAPI::cuda);
  auto cpuQueue = memoryPool::getQueue(nullptr,memoryPool::DeviceAPI::posix);

  {
    auto pd = memoryPool::makeBuffer<int>(20, queue, memoryPool::Where::onDevice);
    auto ph = memoryPool::makeBuffer<int>(20, queue, memoryPool::Where::onHost);
    auto pc = memoryPool::makeBuffer<int>(20, cpuQueue, memoryPool::Where::onDevice);
    assert(20==pd.size());
    assert(20==ph.size());
    assert(20==pc.size());

    memoryPool::copy(ph, pd, queue);
    memoryPool::copy(pd, ph, queue);
    std::cout << "expect 2a 2u 1a 1u" << std::endl;
    memoryPool::dumpStatCUDA();

    {
      auto ph = memoryPool::makeBuffer<int>(20, queue, memoryPool::Where::onHost);
      memoryPool::copy(pd, ph, queue);
    }
    cudaStreamSynchronize(stream);
    std::cout << "expect 2a 2u 2a 1u " << std::endl;
    memoryPool::dumpStatCUDA();
    {
      auto ph = memoryPool::makeBuffer<int>(20, queue, memoryPool::Where::onHost);
      memoryPool::copy(pd, ph, queue);
    }
    std::cout << "expect 2a 2u 2a 1u " << std::endl;
    cudaStreamSynchronize(stream);
    memoryPool::dumpStatCUDA();
  }
  std::cout << "expect 2a 0u 2a 0u " << std::endl;
  cudaStreamSynchronize(stream);
  memoryPool::dumpStatCUDA();

  {
    memoryPool::Deleter devDeleter(std::make_shared<memoryPool::BundleDelete>(queue, memoryPool::Where::onDevice));
    memoryPool::Deleter hosDeleter(std::make_shared<memoryPool::BundleDelete>(queue, memoryPool::Where::onHost));

    auto p0 = memoryPool::makeBuffer<int>(20, devDeleter);
    auto p1 = memoryPool::makeBuffer<double>(20, devDeleter);
    auto p2 = memoryPool::makeBuffer<bool>(20, devDeleter);
    auto p3 = memoryPool::makeBuffer<int>(20, devDeleter);

    {
      auto pd = memoryPool::makeBuffer<int>(40, queue, memoryPool::Where::onDevice);
      assert(40==pd.size());
      assert(20==p0.size());
      p0.reset(pd.release());
      assert(40==p0.size());
      assert(0==pd.size());
      memoryPool::dumpStatCUDA();
    }
    cudaStreamSynchronize(stream);

    auto hp0 = memoryPool::makeBuffer<int>(40, hosDeleter);
    auto hp1 = memoryPool::makeBuffer<double>(20, hosDeleter);
    auto hp2 = memoryPool::makeBuffer<bool>(20, hosDeleter);
    auto hp3 = memoryPool::makeBuffer<int>(20, hosDeleter);

    memoryPool::copy(hp3, p3, queue);
    memoryPool::copy(p0, hp0, queue);
    

    memoryPool::dumpStatCUDA();
  }

  cudaStreamSynchronize(stream);
  memoryPool::dumpStatCUDA();
  memoryPool::shutdownCUDA();

  return 0;
}
