#include "HeterogeneousCore/CUDAUtilities/interface/eigenSOA.h"
#include <Eigen/Dense>

template<int32_t S> 
struct MySOA {

  // we can find a way to avoid this copy/paste???
  static constexpr int32_t stride() { return S; }  

  eigenSOA::ScalarSOA<float,S> a;
  eigenSOA::ScalarSOA<float,S> b;

};

// needed to support cpu and gpu
// can contain multiple independent ASOA
class myDataView {
public:
  static constexpr int32_t S = 256;
  using V = MySOA<S>;

  // various deleted constructor
  // ....

  // proper constructor and setters
  // ....

  // really needed?????
  constexpr V & operator()(int32_t i)  { return data_[i];}
  constexpr V const & operator()(int32_t i) const { return data_[i];}

  constexpr V * data()  { return data_;}
  constexpr V const * data() const { return data_;}


  constexpr int32_t size() const { return n_;}

  V * data_;
  int32_t n_;

};

__global__
void testBasicSOA(float * p) {

  using namespace eigenSOA;
  using V = myDataView::V;

  assert(!isPowerOf2(0));
  assert(isPowerOf2(1));
  assert(isPowerOf2(1024));
  assert(!isPowerOf2(1026));

  using M3 = Eigen::Matrix<float,3,3>;;
  __shared__ eigenSOA::MatrixSOA<M3,64> m;


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

__global__
void copyInA(float const * p, myDataView::V * psoa, int n) {
  using V = myDataView::V;
  int first = threadIdx.x + blockIdx.x*blockDim.x;
  for (auto i=first; i<n; i+=blockDim.x*gridDim.x) {
    auto j = i/V::stride();
    auto k = i%V::stride();
    auto & soa = psoa[j];
    soa.a[k] = p[i];
    soa.b[k] = 0;
  }
}

__global__
void sum(myDataView::V * psoa, int n) {
  using V = myDataView::V;
  int first = threadIdx.x + blockIdx.x*blockDim.x;
  for (auto i=first; i<n; i+=blockDim.x*gridDim.x) {
    auto j = i/V::stride();
    auto k = i%V::stride();
    auto & soa = psoa[j];
    soa.b(k) += soa.a(k);
  }
}


__global__
void sum2(myDataView::V * psoa, int n) {
  using V = myDataView::V;
  int nb = (n+V::stride()-1)/V::stride();
  for (int j=blockIdx.x; j<nb; j+=gridDim.x) {
    auto & soa = psoa[j];
    int kmax = std::min(V::stride(),n - j*V::stride());
    for(int32_t k=threadIdx.x; k<kmax; k+=blockDim.x) {
     soa.b(k) += soa.a(k);
    }
  }
}



#include <random>
#include <cassert>

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
  testBasicSOA<<<1,1024>>>(p_d);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaMemcpy(p,p_d,1024*4,cudaMemcpyDefault));
  cudaCheck(cudaDeviceSynchronize());
#else
  testBasicSOA(p);
#endif

  std::cout << p[0] << std::endl;

  for (int i=0, n=64*3; i<n; ++i)
   assert(p[i]>1.);



  using V = myDataView::V;

  // how many I need???

  constexpr int N = 1000;

  constexpr int asoaSize = (N+V::stride()-1)/V::stride();
  
  V v[asoaSize];

#ifdef __CUDACC__
  V * v_d;
  cudaCheck(cudaMalloc(&v_d,asoaSize*sizeof(V)));
  copyInA<<<asoaSize,V::stride()>>>(p_d,v_d,N);
  cudaCheck(cudaGetLastError());
  sum<<<64,128>>>(v_d,N); // why not...
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaMemcpy(v,v_d,asoaSize*sizeof(V),cudaMemcpyDefault));
#else
  copyInA(p,v,N);
  sum(v,N);
#endif

  std::cout << v[0].a[0] << std::endl;
  std::cout << v[0].b[0] << std::endl;

  for (int i=0, n=N; i<n; ++i) {
    auto j = i/V::stride();
    auto k = i%V::stride();
    assert(p[i]==v[j].a[k]);
    assert(p[i]==v[j].b[k]);
  }
  return 0;



}


