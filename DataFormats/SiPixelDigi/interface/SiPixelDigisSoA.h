#ifndef DataFormats_SiPixelDigi_SiPixelDigisSoA_H
#define DataFormats_SiPixelDigi_SiPixelDigisSoA_H

#include "HeterogeneousCore/CUDAUtilities/interface/eigenSoA.h"

template<int S>
struct SiPixelDigisSoA {

  static constexpr int32_t stride() { return S; }

  eigenSoA::ScalarSoA<uint16_t,S> xx;
  eigenSoA::ScalarSoA<uint16_t,S> yy;
  eigenSoA::ScalarSoA<uint16_t,S> adc;
  eigenSoA::ScalarSoA<uint16_t,S> moduleInd;
  eigenSoA::ScalarSoA<int32_t,S>  clus;

};

#include "HeterogeneousCore/CUDAUtilities/interface/ASoA.h"

using SiPixelDigisASoA = GPU::ASoA<SiPixelDigisSoA<1024>>;

#endif
