#include "HeterogeneousCore/CUDATest/interface/CUDAThing.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include<cmath>

namespace testConvertRecHits{

  __global__
  void
  convertKernel(TrackingRecHit2DSOAView const * viewOnDevice, CUDAThing::View myView) {

    auto const& hh = *viewOnDevice;
    auto first = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 == first)
      printf("converting %d Hits\n",myView.nHits);

    for (int i = first, ni=myView.nHits; i < ni; i += gridDim.x * blockDim.x) {
      if (hh.charge(i) <=0) continue;
      auto xg = hh.xGlobal(i);
      auto yg = hh.yGlobal(i);
      myView.r[i] = std::sqrt(xg*xg+yg*yg); 
      myView.eta[i] = ::sinhf(hh.zGlobal(i)/myView.r[i]);
      myView.phi[i] = std::atan2(yg,xg);
    }
  }


  void convert(TrackingRecHit2DSOAView const * viewOnDevice, CUDAThing::View & myView, cudaStream_t stream) {
     int nHits = myView.nHits;
     if (0==nHits) return; // cuda does not like 0 blocks...
     int threadsPerBlock = 256;
     int blocks = (nHits + threadsPerBlock - 1) / threadsPerBlock;
     convertKernel<<<blocks,threadsPerBlock,0,stream>>>(viewOnDevice,myView);
  }

}

