#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "SimTracker/TrackerHitAssociation/interface/trackerHitAssociationHeterogeneous.h"

__global__
void analyzeHits(TrackingRecHit2DSOAView const* hhp, uint32_t nhits, double * ws) {


  auto const& hh = *hhp;
  auto first = blockIdx.x * blockDim.x + threadIdx.x;

  if (0 == first)
    printf("in analyzeClusterTP\n");

  __shared__ int nh;
  __shared__ double sch;
 
  nh=0; sch=0;
  __syncthreads();

  for (int i = first, ni=nhits; i < ni; i += gridDim.x * blockDim.x) {
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
     printf("in event  : found %f hits, average charge is %f\n",ws[0], ws[1]/ws[0]);  
   }
  }

}
