#include "ClusterSLOnGPU.h"
#include<cassert>
#include<atomic>

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


template<class ForwardIt, class T, class Compare>
__device__
ForwardIt lowerBound(ForwardIt first, ForwardIt last, const T& value, Compare comp)
{
    ForwardIt it;
    auto count = last-first;
 
    while (count > 0) {
        it = first;
        auto step = count / 2;
        it+=step;
        if (comp(*it, value)) {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }
    return first;
}

__global__
void simLink(context const * ddp, uint32_t ndigis, HitsOnGPU const * hhp, ClusterSLGPU const * slp, uint32_t n) {

  constexpr uint16_t InvId=9999; // must be > MaxNumModules
  
  auto const & dd = *ddp;
  auto const & hh = *hhp;
  auto const & sl = *slp;
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (i>ndigis) return;

  auto id = dd.moduleInd_d[i];
  if (InvId==id) return;
  assert(id<2000);

  auto ch = pixelToChannel(dd.xx_d[i], dd.yy_d[i]);
  auto first = hh.hitsModuleStart_d[id];
  auto cl = first + dd.clus_d[i];
  assert(cl<256*2000);
  
  auto rescale = [](uint64_t i, uint64_t n1, uint64_t n2)-> uint32_t {
    return (i*n2)/n1;
  };

  std::array<uint32_t,3> me{{id,ch,0}};

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

  // auto j = std::min(rescale(i,ndigis,n) ,n-1);
  // j = search(me,j);
  auto p = lowerBound(sl.links_d,sl.links_d+n,me,less);
  auto j = p-sl.links_d;
  assert(j>=0);
  j = std::min(int(j),int(n-1));
  if (equal(me,sl.links_d[j])) {
    auto const & l = sl.links_d[j];
    auto const tk = l[2];
    auto old = atomicCAS(&sl.tkId_d[cl],0,tk);
    if (0==old ||tk==old) atomicAdd(&sl.n1_d[cl],1);
    else {
      auto old = atomicCAS(&sl.tkId2_d[cl],0,tk);
      if (0==old ||tk==old) atomicAdd(&sl.n2_d[cl],1);
    }    
  } 
  /*
  else {
    auto const & k=sl.links_d[j];
    auto const & kk = j+1<n ? sl.links_d[j+1] : k;
    printf("digi not found %d:%d closest %d:%d:%d, %d:%d:%d\n",id,ch, k[0],k[1],k[2], kk[0],kk[1],kk[2]);
  }
  */

}


__global__
void dumpLink(int ev, HitsOnGPU const * hhp, uint32_t nhits, ClusterSLGPU const * slp) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>nhits) return;

  auto const & hh = *hhp;
  auto const & sl = *slp;

  printf("HIT: %d %d %d %d %f %f %f %f %d %d %d %d %d\n",ev, i, 
         hh.detInd_d[i], hh.charge_d[i], 
         hh.xg_d[i],hh.yg_d[i],hh.zg_d[i],hh.rg_d[i],hh.iphi_d[i], 
         sl.tkId_d[i],sl.n1_d[i],sl.tkId2_d[i],sl.n2_d[i]
        );

}



namespace clusterSLOnGPU {

  struct CSVHeader {
     CSVHeader() {
      printf("HIT: %s %s %s %s %s %s %s %s %s %s %s %s %s\n", "ev", "ind",
         "det", "charge",	
         "xg","yg","zg","rg","iphi", 
         "tkId","n1","tkId2","n2" 
        );
     }

  };
  CSVHeader csvHeader;

  std::atomic<int> evId(0);

  void wrapper(context const & dd, uint32_t ndigis, HitsOnGPU const & hh, uint32_t nhits, ClusterSLGPU const & sl, uint32_t n) {
    
    int ev = ++evId;
    int threadsPerBlock = 256;
    int blocks = (ndigis + threadsPerBlock - 1) / threadsPerBlock;

    assert(sl.me_d);
    simLink<<<blocks, threadsPerBlock, 0, dd.stream>>>(dd.me_d,ndigis, hh.me_d, sl.me_d,n);
    blocks = (nhits + threadsPerBlock - 1) / threadsPerBlock;
    dumpLink<<<blocks, threadsPerBlock, 0, dd.stream>>>(ev, hh.me_d, nhits, sl.me_d);
    cudaCheck(cudaGetLastError());

  }

}
