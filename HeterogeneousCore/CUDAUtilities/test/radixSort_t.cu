#include "HeterogeneousCore/CUDAUtilities/interface/radixSort.h"

template<typename T>
__global__
void radixSortMultiWrapper(T * v, uint16_t * index, uint32_t * offsets) {
  radixSortMulti(v,index,offsets);
}

#include "cuda/api_wrappers.h"

#include <iomanip>
#include <memory>
#include <algorithm>
#include <chrono>
#include<random>


#include<cassert>
#include<iostream>
#include<limits>


template<typename T>
void go() {

std::mt19937 eng;
// std::mt19937 eng2;
std::uniform_int_distribution<T> rgen(std::numeric_limits<T>::min(),std::numeric_limits<T>::max());


  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;

  if (cuda::device::count() == 0) {
	std::cerr << "No CUDA devices on this system" << "\n";
	exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get(); 

  constexpr int blocks=10;
  constexpr int blockSize = 256*32;
  constexpr int N=blockSize*blocks;
  T v[N];
  uint16_t ind[N];

  std::cout << "Will sort " << N << " 'ints' of size " << sizeof(T) << std::endl;


  for (int i=0; i<50; ++i) {

    if (i==49) { 
        for (long long j = 0; j < N; j++) v[j]=0;
    } else if (i>30) {
    for (long long j = 0; j < N; j++) v[j]=rgen(eng);
    } else {
      long long imax = (i<15) ? std::numeric_limits<T>::max() +1LL : 255;
      for (long long j = 0; j < N; j++) {
        v[j]=(j%imax); if(j%2 && i%2) v[j]=-v[j];
      }
    }

  uint32_t offsets[blocks+1];
  offsets[0]=0;
  for (int j=1; j<blocks+1; ++j) offsets[j] = offsets[j-1]+blockSize;


  std::random_shuffle(v,v+N);
  auto v_d = cuda::memory::device::make_unique<T[]>(current_device, N);
  auto ind_d = cuda::memory::device::make_unique<uint16_t[]>(current_device, N);
  auto off_d = cuda::memory::device::make_unique<uint32_t[]>(current_device, blocks+1);

  cuda::memory::copy(v_d.get(), v, N*sizeof(T));
  cuda::memory::copy(off_d.get(), offsets, 4*(blocks+1));


   int threadsPerBlock =256;
   int blocksPerGrid = blocks;
   delta -= (std::chrono::high_resolution_clock::now()-start);
   cuda::launch(
                radixSortMultiWrapper<T>,
                { blocksPerGrid, threadsPerBlock },
                v_d.get(),ind_d.get(),off_d.get()
        );


//  cuda::memory::copy(v, v_d.get(), 2*N);
   cuda::memory::copy(ind, ind_d.get(), 2*N);

   delta += (std::chrono::high_resolution_clock::now()-start);

  if (32==i) {
    std::cout << v[ind[0]] << ' ' << v[ind[1]] << ' ' << v[ind[2]] << std::endl;
    std::cout << v[ind[3]] << ' ' << v[ind[10]] << ' ' << v[ind[blockSize-1000]] << std::endl;
    std::cout << v[ind[blockSize/2-1]] << ' ' << v[ind[blockSize/2]] << ' ' << v[ind[blockSize/2+1]] << std::endl;
  }
  for (int ib=0; ib<blocks; ++ib)
  for (auto i = offsets[ib]+1; i < offsets[ib+1]; i++) {
      auto a = v+offsets[ib];
   // assert(!(a[ind[i]]<a[ind[i-1]]));
     if (a[ind[i]]<a[ind[i-1]])
      std::cout << ib << " not ordered at " << ind[i] << " : "
  		<< a[ind[i]] <<' '<< a[ind[i-1]] << std::endl;
  }
 }  // 50 times
     std::cout <<"cuda computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/50.
              << " ms" << std::endl;
}


int main() {

  go<int16_t>();
  go<int32_t>();
  return 0;
}
