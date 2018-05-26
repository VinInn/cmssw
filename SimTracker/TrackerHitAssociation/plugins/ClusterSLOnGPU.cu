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
void simLink(context const * ddp, uint32_t ndigis, HitsOnGPU const * hhp, ClusterSLGPU const * slp, uint32_t n) {

  constexpr uint16_t InvId=9999; // must be > MaxNumModules
  
  auto const & dd = *ddp;
  auto const & hh = *hhp;
  auto const & sl = *slp;
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (i<256*ClusterSLGPU::MaxNumModules) {
    sl.tkId_d[i]=0; sl.tkId2_d[i]=0; sl.n1_d[i]=0; sl.n2_d[i]=0;
  }
  __syncthreads();
  
  if (i>ndigis) return;

  auto id = dd.moduleInd_d[i];
  if (InvId==id) return;
  assert(id<2000);

  auto ch = pixelToChannel(dd.xx_d[i], dd.yy_d[i]);
  auto first = hh.hitsModuleStart_d[id];
  auto cl = first + dd.clus_d[i];
  assert(cl<256*2000);
  
  std::array<uint32_t,3> me{{id,ch,0}};
  auto j = std::min(i,n-1);

  auto less = [](std::array<uint32_t,3> const & a, std::array<uint32_t,3> const & b)->bool {
     return a[0]<b[0] || ( !(b[0]<a[0]) && a[1]<b[1]); // in this context we do care of [2] 
  };

  auto equal = [](std::array<uint32_t,3> const & a, std::array<uint32_t,3> const & b)->bool {
     return a[0]==b[0] && a[1]==b[1]; // in this context we do care of [2]
  };


  auto search = [&](auto const & a, unsigned int j) {
    auto const & v = sl.links_d;
    auto const s = n;
    if (!less(a,v[j])) {while( j<s && (!less(a,v[++j])) ){} return j-1;}
    if (less(a,v[j]))  {while( j>0 &&   less(a,v[--j])  ){} return j;}
    return j;
  };

  j = search(me,j);
  assert(j<n);
  if (equal(me,sl.links_d[j])) {
    printf("digi found %d:%d\n",id,ch);

  } else {
    auto const & k=sl.links_d[j];
    auto const & kk = j+1<n ? sl.links_d[j+1] : k;
    printf("digi not found %d:%d closest %d:%d:%d, %d:%d:%d\n",id,ch, k[0],k[1],k[2], kk[0],kk[1],kk[2]);
  }

}

namespace clusterSLOnGPU {

  void wrapper(context const & dd, uint32_t ndigis, HitsOnGPU const & hh, ClusterSLGPU const & sl, uint32_t n) {

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    assert(sl.me_d);
    simLink<<<blocks, threadsPerBlock, 0, dd.stream>>>(dd.me_d,ndigis, hh.me_d, sl.me_d,n);
    cudaCheck(cudaGetLastError());

  }

}
