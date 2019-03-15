#ifndef HeterogeneousCore_CUDAUtilities_ASOA_H
#define HeterogeneousCore_CUDAUtilities_ASOA_H

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include<cstdint>
#include<algorithm>


namespace GPU {  


constexpr bool isPowerOf2(uint32_t v) {
    return v && !(v & (v - 1));
}


template<typename SOA, int32_t S=SOA::stride()>
class ASOA {
public:

  using data_type = SOA;

  static constexpr int32_t stride() { return S; }
  static_assert(isPowerOf2(S),"stride not a power of 2");

  // given a capacity return the required size of the data array
  // given the size will return the number of filled SOAs.
  static constexpr int32_t dataSize(int32_t icapacity) {
     return (icapacity+stride()-1)/stride();
  }

  // given a capacity return the required size in byte of the data array
  static constexpr int32_t dataBytes(int32_t icapacity) {
      return dataSize(icapacity)*sizeof(data_type);
  }

  struct Indices{int32_t j,k;};

  // return the index of the SOA and the index of the element in it
  // in c++17:  auto [j,k] = asoa.indeces(i); auto & soa = asoa[j];  soa.x[k];
  //static constexpr
  //auto indices(int32_t i) { return std::make_tuple(i/stride(), i%stride());}
  // in cuda: auto jk =  asoa.indeces(i); auto & soa = asoa[jk.j];  soa.x[jk.k];
  static constexpr
  auto indices(int32_t i) { return Indices{i/stride(), i%stride()};}

  __device__ __host__
  void construct(int32_t icapacity, SOA * idata) {
    m_size = 0;
    m_capacity = icapacity;
    m_data = idata;
  }


  inline constexpr bool empty() const { return 0 == m_size; }
  inline constexpr bool full() const { return m_capacity == m_size; }
  inline constexpr void clear() { m_size = 0; }
  inline constexpr auto size() const { return m_size; }
  inline constexpr auto capacity() const { return m_capacity; }

  // these manage the SOA themselves
  inline constexpr SOA & operator[](int32_t i) { return m_data[i]; }
  inline constexpr const SOA& operator[](int32_t i) const { return m_data[i]; }
  inline constexpr SOA * & data() { return m_data; }
  inline constexpr SOA const * data() const { return m_data; }


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

  SOA * m_data;
};


} // GPU

#endif
