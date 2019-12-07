#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "SimTracker/TrackerHitAssociation/interface/trackerHitAssociationHeterogeneous.h"

__device__ int nev=0;

__global__
void analyzeClusterTP(SiPixelDigisCUDA::DeviceConstView const* dd, uint32_t ndigis, TrackingRecHit2DSOAView const* hhp, uint32_t nhits, trackerHitAssociationHeterogeneous::ClusterSLView clusSL, double * ws) {


  auto const& hh = *hhp;
  auto first = blockIdx.x * blockDim.x + threadIdx.x;

  if (0 == first && 0==nev)
    printf("in analyzeClusterTP\n");

  __shared__ int nh;
  __shared__ double sch;
 
  nh=0; sch=0;
  __syncthreads();

  for (int i = first; i < nhits; i += gridDim.x * blockDim.x) {
    if (hh.charge(i) <=0) printf("a hit with zero charge?\n");
    atomicAdd(&nh,1);
    atomicAdd(&sch,(double)(hh.charge(i)));
  }

  __syncthreads();

  if(0==threadIdx.x) {
   atomicAdd(&ws[0],nh);
   atomicAdd(&ws[1],sch);
   atomicAdd(&ws[9],1);
   if (gridDim.x==ws[9]) {
     atomicAdd(&nev,1);
     printf("in event %d, found %f hits, average charge is %f\n",nev,ws[0], ws[1]/ws[0]);  
   }
  }

}
