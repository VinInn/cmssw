#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memoryPool.h"

namespace testTrackingRecHit2D {

  void runKernels(TrackingRecHit2DSOAView* hits);

}

int main() {
  cms::cudatest::requireDevices();
  memoryPool::initCUDA();

  auto nHits = 200;

  {
  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  assert(stream);
  auto queue = memoryPool::getQueue(stream, memoryPool::DeviceAPI::cuda);
  assert(stream==queue->m_stream);


  // inner scope to deallocate memory before destroying the stream
  {
    TrackingRecHit2DGPU tkhit(nHits, false, 0, nullptr, nullptr, queue);
    testTrackingRecHit2D::runKernels(tkhit.view());

    TrackingRecHit2DGPU tkhitPhase2(nHits, true, 0, nullptr, nullptr, queue);
    testTrackingRecHit2D::runKernels(tkhitPhase2.view());

    memoryPool::dumpStatCUDA();

    TrackingRecHit2DHost tkhitH(nHits, false, 0, nullptr, nullptr, queue, &tkhit);

    cudaCheck(cudaStreamSynchronize(stream));
    memoryPool::dumpStatCUDA();

    std::cout <<"\n\nTransferred to host\n" << std::endl;

    assert(tkhitH.view());
    assert(tkhitH.view()->nHits() == unsigned(nHits));
    assert(tkhitH.view()->nMaxModules() == phase1PixelTopology::numberOfModules);

    TrackingRecHit2DHost tkhitHPhase2(nHits, true, 0, nullptr, nullptr, queue, &tkhitPhase2);
    cudaCheck(cudaStreamSynchronize(stream));
    assert(tkhitHPhase2.view());
    assert(tkhitHPhase2.view()->nHits() == unsigned(nHits));
    assert(tkhitHPhase2.view()->nMaxModules() == phase2PixelTopology::numberOfModules);

    memoryPool::dumpStatCUDA();
  }

  cudaCheck(cudaStreamSynchronize(stream));
  memoryPool::dumpStatCUDA();

  cudaCheck(cudaStreamDestroy(stream));

  memoryPool::dumpStatCUDA();
  memoryPool::shutdownCUDA();
  }

  std::cout << "\n\nDone with CUDA now CPU\n" << std::endl;

  {
   memoryPool::initCPU();
  
    auto queue = memoryPool::getQueue(nullptr, memoryPool::DeviceAPI::posix);
    assert(nullptr==queue->m_stream);
    std::cout << "on CPU" << std::endl;
    queue->m_devicePool->dumpStat();

  {
    TrackingRecHit2DCPU tkhit(nHits, false, 0, nullptr, nullptr, queue);
    assert(tkhit.view());
    assert(tkhit.view()->nHits() == unsigned(nHits));
    assert(tkhit.view()->nMaxModules() == phase1PixelTopology::numberOfModules);
    std::cout << "on CPU" << std::endl;
    queue->m_devicePool->dumpStat();
  }
  std::cout << "on CPU" << std::endl;
  queue->m_devicePool->dumpStat();

  memoryPool::shutdownCPU();
  }

  return 0;
}
