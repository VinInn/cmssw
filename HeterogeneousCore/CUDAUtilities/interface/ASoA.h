#ifndef HeterogeneousCore_CUDAUtilities_ASoA_H
#define HeterogeneousCore_CUDAUtilities_ASoA_H

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include<cstdint>
#include<algorithm>
#include <cstdlib>

namespace GPU {  


constexpr bool isPowerOf2(uint32_t v) {
    return v && !(v & (v - 1));
}

struct ExternalStorage {
   template<typename T>
   __host__ __device__
   void alloc(T **, int){}
   __host__ __device__
   void dealloc(void *) {}
};

struct CPUStorage {
   template<typename T>
   void alloc(T ** p, int n){
       (*p) = (T*)malloc(n);
   }
   void dealloc(void *p ) { free(p); }
};


// storage is now in the type...
// will cause trouble even if being stride template argument all client will have to template...
template<typename SoA, int32_t S=SoA::stride(), typename Storage=ExternalStorage>
class ASoA : private Storage {
public:

  ASoA() = default;
  ASoA(ASoA const &) = delete;
  ASoA(ASoA &&) = delete;
  ASoA operator=(ASoA const &) = delete;
  ASoA operator=(ASoA &&) = delete;

  ~ASoA() { this->Storage::dealloc(m_data);}

  using data_type = SoA;

  static constexpr int32_t stride() { return S; }
  static_assert(isPowerOf2(S),"stride not a power of 2");

  // given a capacity return the required size of the data array
  // given the size will return the number of filled SoAs.
  static constexpr int32_t dataSize(int32_t icapacity) {
     return (icapacity+stride()-1)/stride();
  }

  // given a capacity return the required size in byte of the data array
  static constexpr int32_t dataBytes(int32_t icapacity) {
      return dataSize(icapacity)*sizeof(data_type);
  }

  struct Indices{int32_t j,k;};

  // return the index of the SoA and the index of the element in it
  // in c++17:  auto [j,k] = asoa.indeces(i); auto & soa = asoa[j];  soa.x[k];
  //static constexpr
  //auto indices(int32_t i) { return std::make_tuple(i/stride(), i%stride());}
  // in cuda: auto jk =  asoa.indeces(i); auto & soa = asoa[jk.j];  soa.x[jk.k];
  static constexpr
  auto indices(int32_t i) { return Indices{i/stride(), i%stride()};}

  // __device__ __host__
  void construct(int32_t icapacity, SoA * idata) {
    m_size = 0;
    m_capacity = icapacity;
    m_data = idata;
    if (nullptr == m_data) this->Storage::alloc(&m_data,dataBytes(m_capacity));
    assert(m_data);  // ok throw bad_alloc...
  }


  inline constexpr bool empty() const { return 0 == m_size; }
  inline constexpr bool full() const { return m_capacity == m_size; }
  inline constexpr void clear() { m_size = 0; }
  inline constexpr auto size() const { return m_size; }
  inline constexpr auto capacity() const { return m_capacity; }

  // these manage the SoA themselves
  inline constexpr SoA & operator[](int32_t i) { return m_data[i]; }
  inline constexpr const SoA& operator[](int32_t i) const { return m_data[i]; }
  inline constexpr SoA * & data() { return m_data; }
  inline constexpr SoA const * data() const { return m_data; }


  __device__
  int32_t addOne() {
    auto previousSize = atomicAdd(&m_size, 1);
    if (previousSize < m_capacity) {
      return previousSize;
    } else {
      atomicSub(&m_size, 1);
      return -1;
    }
  }

private:
  int32_t m_size=0;
  int32_t m_capacity;

  SoA * m_data;
};


} // GPU

#endif
