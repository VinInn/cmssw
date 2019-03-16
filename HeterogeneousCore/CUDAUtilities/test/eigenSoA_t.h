#include "HeterogeneousCore/CUDAUtilities/interface/eigenSoA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/ASoA.h"

#include <Eigen/Dense>

template<int32_t S> 
struct MySoA {

  // we can find a way to avoid this copy/paste???
  static constexpr int32_t stride() { return S; }  

  eigenSoA::ScalarSoA<float,S> a;
  eigenSoA::ScalarSoA<float,S> b;

};

using V = MySoA<128>;
using AVE = GPU::ASoA<V>;
using AVS = GPU::ASoA<V,V::stride(),GPU::CPUStorage>;


__global__
void testBasicSoA(float * p) {

  using namespace eigenSoA;

  assert(!isPowerOf2(0));
  assert(isPowerOf2(1));
  assert(isPowerOf2(1024));
  assert(!isPowerOf2(1026));

  using M3 = Eigen::Matrix<float,3,3>;;
  __shared__ eigenSoA::MatrixSoA<M3,64> m;


  int first = threadIdx.x + blockIdx.x*blockDim.x;
  if(0==first) printf("before %f\n",p[0]);

   // a silly game...
   int n=64;
   for (int i=first; i<n; i+=blockDim.x*gridDim.x) {   
     m[i].setZero();
     m[i](0,0) = p[i];
     m[i](1,1) = p[i+64];
     m[i](2,2) = p[i+64*2];
   }
   __syncthreads(); // not needed

   for (int i=first; i<n; i+=blockDim.x*gridDim.x)
     m[i] = m[i].inverse().eval();
   __syncthreads();

   for (int i=first; i<n; i+=blockDim.x*gridDim.x) {
     p[i] = m[63-i](0,0);
     p[i+64] = m[63-i](1,1);
     p[i+64*2] = m[63-i](2,2);
   }

  if(0==first) printf("after %f\n",p[0]);

}

template<typename AV>
__global__
void copyInA(float const * p, AV * pasoa, int n) {
  auto & asoa = *pasoa;
  int first = threadIdx.x + blockIdx.x*blockDim.x;
  for (auto i=first; i<n; i+=blockDim.x*gridDim.x) {
    auto ind = asoa.addOne();
    if (ind<0) return;
    assert(ind<asoa.capacity());
    assert(ind<n);
    auto jk = AV::indices(ind);
    assert(jk.k<AV::stride());
    auto & soa = asoa[jk.j];
    soa.a[jk.k] = p[ind]; // make it deterministic....
    soa.b[jk.k] = 0;
  }
}

template<typename AV>
__global__
void sum(AV * pasoa) {
  auto & asoa = *pasoa;
  int32_t first = threadIdx.x + blockIdx.x*blockDim.x;
  for (auto i=first,n=asoa.size(); i<n; i+=blockDim.x*gridDim.x) {
    assert(i<asoa.capacity());
    auto jk = AV::indices(i);
    assert(jk.k<AV::stride());
    auto & soa = asoa[jk.j];
    soa.b[jk.k] += soa.a[jk.k];
  }
}



#include <random>
#include <cassert>
#include <memory>

#ifdef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#endif

#include<iostream>

int main() {

#ifdef __CUDACC__
  exitSansCUDADevices();
#endif

  float p[1024];

  std::uniform_real_distribution<float> rgen(0.01,0.99);
  std::mt19937 eng;

  for (auto & r : p) r = rgen(eng);
  for (int i=0, n=64*3; i<n; ++i)
   assert(p[i]>0 && p[i]<1.);

  std::cout << p[0] << std::endl;
#ifdef __CUDACC__
  float * p_d;
  cudaCheck(cudaMalloc(&p_d,1024*4));
  cudaCheck(cudaMemcpy(p_d,p,1024*4,cudaMemcpyDefault));
  testBasicSoA<<<1,1024>>>(p_d);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaMemcpy(p,p_d,1024*4,cudaMemcpyDefault));
  cudaCheck(cudaDeviceSynchronize());
#else
  testBasicSoA(p);
#endif

  std::cout << p[0] << std::endl;

  for (int i=0, n=64*3; i<n; ++i)
   assert(p[i]>1.);


// ASoA....

  // how many I need???

  const int N = 1000;

  // auto v_data = std::make_unique<AV::data_type[]>(AV::dataSize(N));
  AVS av; av.construct(N,nullptr);
  assert(N == av.capacity());
  assert(av.empty());
  // assert(av.data()==v_data.get());

  std::cout << "number of buckets " << AVS::dataSize(N) << " " << " capacity " << av.capacity() << " stride " << AVS::stride() << std::endl;
  std::cout << "size of data array " << AVS::dataBytes(N) << std::endl;
  assert(av.capacity()<=AVS::dataSize(N)*AVS::stride());

#ifdef __CUDACC__
  AVE::data_type * v_d;
  cudaCheck(cudaMalloc(&v_d,AVE::dataBytes(N)));
  AVE av_h; av_h.construct(N,v_d);  // hold device data pointer...
  AVE * av_d;
  cudaCheck(cudaMalloc(&av_d,sizeof(AVE)));
  cudaCheck(cudaMemcpy(av_d,&av_h,sizeof(AVE),cudaMemcpyDefault));
  copyInA<<<AVE::dataSize(N),AVE::stride()>>>(p_d,av_d,N);
  cudaCheck(cudaGetLastError());
  sum<<<64,128>>>(av_d); // why not...
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaMemcpy(&av,av_d,sizeof(int32_t),cudaMemcpyDefault));  // this can be async...
  cudaCheck(cudaMemcpy(av.data(),av_h.data(),AVE::dataBytes(N),cudaMemcpyDefault));
#else
  copyInA(p,&av,N);
  sum(&av);
#endif

  assert(N == av.capacity());
  //  assert(av.data()==v_data.get());
  assert(av.size()==N);

  std::cout << av[0].a[0] << std::endl;
  std::cout << av[0].b[0] << std::endl;

  for (int i=0, n=N; i<n; ++i) {
    auto j = i/AVS::stride();
    auto k = i%AVS::stride();
    assert(p[i]==av[j].a[k]);
    assert(p[i]==av[j].b[k]);
  }


#ifdef __CUDACC__
  cudaCheck(cudaFree(av_d));
  cudaCheck(cudaFree(v_d));
#endif

  std::cout << "END" << std::endl;
  return 0;



}


