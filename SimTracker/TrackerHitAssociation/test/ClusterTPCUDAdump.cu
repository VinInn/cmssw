#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "SimTracker/TrackerHitAssociation/interface/trackerHitAssociationHeterogeneous.h"

__global__
void analyzeClusterTP(SiPixelDigisCUDA::DeviceConstView const* dd, uint32_t ndigis, TrackingRecHit2DSOAView const* hhp, uint32_t nhits, trackerHitAssociationHeterogeneous::ClusterSLView clusSL) {


  auto const& hh = *hhp;
  auto first = blockIdx.x * blockDim.x + threadIdx.x;

  if (0 == first)
    printf("in analyzeClusterTP\n");

  for (int i = first; i < nhits; i += gridDim.x * blockDim.x) {
    if (hh.charge(i) <=0) printf("a hit with zero charge?\n");
  }



}
