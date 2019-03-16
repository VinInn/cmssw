#ifndef DataFormats_SiPixelDigi_SiPixelDigisSoA_H
#define DataFormats_SiPixelDigi_SiPixelDigisSoA_H

#include "HeterogeneousCore/CUDAUtilities/interface/eigenSOA.h"

template<int S>
struct SiPixelDigisSaA {

  static constexpr int32_t stride() { return S; }

  eigenSOA::ScalarSOA<uint16_t,S> xx;
  eigenSOA::ScalarSOA<uint16_t,S> yy;
  eigenSOA::ScalarSOA<uint16_t,S> adc;
  eigenSOA::ScalarSOA<uint16_t,S> moduleInd;
  eigenSOA::ScalarSOA<int32_t,S>  clus;

};


#endif
