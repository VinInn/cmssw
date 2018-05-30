#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <cmath>

#include<iostream>

  constexpr
  short phi2short(float x) {
    constexpr float p2i = ( (int)(std::numeric_limits<short>::max())+1 )/M_PI;
    return std::round(x*p2i);
  }


  void  slidingWindow(uint16_t mysize, uint16_t mymin,uint16_t mymax) {
    uint16_t imax = std::numeric_limits<uint16_t>::max();
    uint16_t offset = (mymin>mymax) ? imax-(mysize-1) : 0;
    int n=0;
    for (uint16_t i = mymin+offset; i!=mymax; i++) {
      assert(i<=imax);
      uint16_t k = (i>mymax) ? i-offset : i;
      assert(k<mysize);
      assert(k>=mymin || k<mymax);
      n++;
    }
    int tot = (mymin>mymax) ? (mysize-mymin)+mymax : mymax-mymin;
    assert(n==tot);
  };


std::pair<int,int> findPhiLimits(int16_t phiMe, int16_t * iphi, uint16_t * index, uint16_t size, float phiCut) {

  // auto idelta = phicut * float(size)/(2.f*float(M_PI));
  auto iphicut = phi2short(phiCut);
  assert(iphicut>0);
  
  // find extreemes in top
  int16_t minPhi = phiMe-iphicut;
  int16_t maxPhi = phiMe+iphicut;

  std::cout << "\n phi min/max " << phiMe << ' ' << minPhi << ' ' << maxPhi << std::endl;

  // guess and adjust
  auto findLimit = [&](int16_t mPhi) { 
    int jm = float(0.5f*size)*(1.f+float(mPhi)/float(std::numeric_limits<short>::max()));
    std::cout << "jm for " << mPhi << ' ' << jm << std::endl;
    jm = std::min(size-1,std::max(0,jm));
    bool notDone=true;
    while(mPhi<iphi[index[--jm]]){notDone=false;}
    if (notDone) while(mPhi>iphi[index[++jm]]){}
    jm = std::min(size-1,std::max(0,jm));
    return jm;
  };
 
  auto jmin = findLimit(minPhi);  
  auto jmax = findLimit(maxPhi);
 
  std::cout << "j min/max " << jmin << ' ' << jmax << std::endl;
  std::cout << "found min/max " << iphi[index[jmin]] << ' ' << iphi[index[jmax]] << std::endl;  
  std::cout << "found min/max +1 " << iphi[index[jmin+1]] << ' ' << iphi[index[jmax+1]] << std::endl;
    std::cout << "found min/max -1 " << iphi[index[jmin-1]] << ' ' << iphi[index[jmax-1]] << std::endl;

 slidingWindow(size,jmin,jmax);
 return std::make_pair(jmin,jmax);
}


#include <algorithm>
#include <chrono>
#include<random>

int main() {

  std::mt19937 eng;
  std::uniform_int_distribution<int16_t> rgen(std::numeric_limits<int16_t>::min(),std::numeric_limits<int16_t>::max());

  constexpr int N=3245;
  int16_t v[N];
  uint16_t ind[N];
  for (int j = 0; j < N; j++) { v[j]=rgen(eng); ind[j]=j;}
  std::sort(ind,ind+N,[&](auto i, auto j){return v[i]<v[j];});
  for (int j = 1; j < N; j++) { assert(v[ind[j-1]]<=v[ind[j]]);}

  std::cout << "size " << N << std::endl;
  findPhiLimits(0, v, ind, N, 0.2); 
  findPhiLimits(-15000, v, ind, N, 0.2);
  findPhiLimits(15000, v, ind, N, 0.2);
  findPhiLimits(-32000, v, ind, N, 0.2);
  findPhiLimits(32000, v, ind, N, 0.2);



  return 0;
}
