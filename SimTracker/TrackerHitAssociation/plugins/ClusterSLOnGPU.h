#ifndef SimTracker_TrackerHitAssociation_plugins_ClusterSLOnGPU_h
#define SimTracker_TrackerHitAssociation_plugins_ClusterSLOnGPU_h

#include <cuda_runtime.h>
#include <cuda/api_wrappers.h>

#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"

#include "trackerHitAssociationHeterogeneousProduct.h"

namespace clusterSLOnGPU {

  using ClusterSLGPU = trackerHitAssociationHeterogeneousProduct::ClusterSLGPU;
  using GPUProduct   = trackerHitAssociationHeterogeneousProduct::GPUProduct;

  using HitsOnGPU    = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
  using HitsOnCPU    = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;

  using Clus2TP = ClusterSLGPU::Clus2TP;

  class Kernel {
  public:
    Kernel(cuda::stream_t<>& stream, bool dump);
    ~Kernel() {deAlloc();}
    void algo(SiPixelDigisCUDA const & dd, uint32_t ndigis, HitsOnCPU const & hh, uint32_t nhits, uint32_t n, cuda::stream_t<>& stream);
    GPUProduct getProduct() { return GPUProduct{slgpu.me_d};}
    
  private:
     void alloc(cuda::stream_t<>& stream);
     void deAlloc(); 
     void zero(cudaStream_t stream);

  public:
     ClusterSLGPU slgpu; 
     bool doDump;
  };
}

#endif // SimTracker_TrackerHitAssociation_plugins_ClusterSLOnGPU_h
