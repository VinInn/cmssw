#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisSOA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisSOA_h

#include "HeterogeneousCore/CUDAUtilities/interface/eigenSOA.h"

template<int S>
struct SiPixelDigisSOA {

  static constexpr int32_t stride() { return S; }

  eigenSOA::ScalarSOA<uint16_t,S> xx;
  eigenSOA::ScalarSOA<uint16_t,S> yy;
  eigenSOA::ScalarSOA<uint16_t,S> adc;
  eigenSOA::ScalarSOA<uint16_t,S> moduleInd;
  eigenSOA::ScalarSOA<int32_t,S>  clus;

};



endif
