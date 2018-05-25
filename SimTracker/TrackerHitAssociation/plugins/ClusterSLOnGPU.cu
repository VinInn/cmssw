#include "ClusterSLOnGPU.h"
#include<cassert>

/*
struct ClusterSLGPU {
 ClusterSLGPU(){alloc();}
 void alloc();

 ClusterSLGPU * me_d;
 std::array<uint32_t,3> * links_d;
 uint32_t * tkId_d;
 uint32_t * tkId2_d;
 uint32_t * n1_d;
 uint32_t * n2_d;

 static constexpr uint32_t MAX_DIGIS = 2000*150;
 static constexpr uint32_t MaxNumModules = 2000;

};
*/

__global__
void simLink(context const * ddp,HitsOnGPU const * hhp, ClusterSLGPU const * slp, uint32_t n) {
  
  auto const & dd = *ddp;
  auto const & hh = *hhp;
  auto const & sl = *slp;
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (i<256*ClusterSLGPU::MaxNumModules) {
    sl.tkId_d[i]=0; sl.tkId2_d[i]=0; sl.n1_d[i]=0; sl.n2_d[i]=0;
  }
  __syncthreads();
  
  if (i>n) return;

  auto ch = pixelToChannel(dd.xx_d[i], dd.yy_d[i]);
  auto id = dd.moduleInd_d[i];
  assert(id<2000);
  auto first = hh.hitsModuleStart_d[id];
  auto cl = first + dd.clus_d[i];
  assert(cl<256*2000);

}

namespace clusterSLOnGPU {

  void wrapper(context const & dd, HitsOnGPU const & hh, ClusterSLGPU const & sl, uint32_t n) {

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    assert(sl.me_d);
    simLink<<<blocks, threadsPerBlock, 0, dd.stream>>>(dd.me_d,hh.me_d,sl.me_d,n);
    cudaCheck(cudaGetLastError());

  }

}
