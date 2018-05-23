#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <cmath>

#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

namespace gpuPixelRecHits {

  // later from Math
  constexpr
  short phi2short(float x) {
    constexpr float p2i = ( (int)(std::numeric_limits<short>::max())+1 )/M_PI;
    return std::round(x*p2i);
  }



constexpr
std::pair<int,int> findPhiLimits(int16_t phiMe, int16_t * iphi, uint16_t * index, uint16_t size, float phiCut) {

  auto iphicut = phi2short(phiCut);
  assert(iphicut>0);
  
  // find extreemes in top
  int16_t minPhi = phiMe-iphicut;
  int16_t maxPhi = phiMe+iphicut;

  // std::cout << "\n phi min/max " << phiMe << ' ' << minPhi << ' ' << maxPhi << std::endl;

  // guess and adjust
  auto findLimit = [&](int16_t mPhi) { 
    int jm = float(0.5f*size)*(1.f+float(mPhi)/float(std::numeric_limits<short>::max()));
    // std::cout << "jm for " << mPhi << ' ' << jm << std::endl;
    jm = std::min(size-1,std::max(0,jm));
    bool notDone=true;
    while(mPhi<iphi[index[--jm]]){notDone=false;}
    if (notDone) while(mPhi>iphi[index[++jm]]){}
    jm = std::min(size-1,std::max(0,jm));
    return jm;
  };
 
  auto jmin = findLimit(minPhi);  
  auto jmax = findLimit(maxPhi);
 
  /*
  std::cout << "j min/max " << jmin << ' ' << jmax << std::endl;
  std::cout << "found min/max " << iphi[index[jmin]] << ' ' << iphi[index[jmax]] << std::endl;  
  std::cout << "found min/max +1 " << iphi[index[jmin+1]] << ' ' << iphi[index[jmax+1]] << std::endl;
  std::cout << "found min/max -1 " << iphi[index[jmin-1]] << ' ' << iphi[index[jmax-1]] << std::endl;
  */

 return std::make_pair(jmin,jmax);
}


__global__
void getDoublets(uint16_t * iphi, uint16_t * index, uint32_t * offsets, float phiCut) {

  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=offsets[9]) return; // get rid of last layer

  assert(offsets[0]=0);
  int top = (i>offsets[5]) ? 5: 0;
  while (i>=offsets[++top]){};
  assert(top<10);
  auto bottom = layer-1;
  if (bottom==3 || bottom==6) return; // do not have UP... (9 we got rid already)

  assert(index[i]<offsets[top]-offsets[bottom]);

  int16_t phiMe = iphi[offsets[bottom]+index[i]];
  
  auto jLimits = findLimits(phiMe, iphi+offsets[top],index+offsets[top],size,phicut);
 
}

