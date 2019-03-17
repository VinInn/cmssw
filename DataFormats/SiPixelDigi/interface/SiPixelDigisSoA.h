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
using SiPixelDigisASoAonCPU = GPU::ASoA<SiPixelDigisSoA<1024>,1024,GPU::CPUStorage>;

// is this really needed?
// seems handy
template<typename ASOA>
class SiPixelDigisSoAView {
public:

  constexpr auto & xx(int32_t i) { constexpr auto jk = ASOA::indices(i); return asoa()[jk.j].xx[jk.k];} 
  constexpr auto xx(int32_t i) const { constexpr auto jk = ASOA::indices(i); return asoa()[jk.j].xx[jk.k];}
  constexpr auto & yy(int32_t i) { constexpr auto jk = ASOA::indices(i); return asoa()[jk.j].yy[jk.k];}
  constexpr auto yy(int32_t i) const { constexpr auto jk = ASOA::indices(i); return asoa()[jk.j].yy[jk.k];}
  constexpr auto & adc(int32_t i) { constexpr auto jk = ASOA::indices(i); return asoa()[jk.j].adc[jk.k];}
  constexpr auto adc(int32_t i) const { constexpr auto jk = ASOA::indices(i); return asoa()[jk.j].adc[jk.k];}
  constexpr auto & moduleInd(int32_t i) { constexpr auto jk = ASOA::indices(i); return asoa()[jk.j].moduleInd[jk.k];}
  constexpr auto moduleInd(int32_t i) const { constexpr auto jk = ASOA::indices(i); return asoa()[jk.j].moduleInd[jk.k];}
  constexpr auto & clus(int32_t i) { constexpr auto jk = ASOA::indices(i); return asoa()[jk.j].clus[jk.k];}
  constexpr auto clus(int32_t i) const { constexpr auto jk = ASOA::indices(i); return asoa()[jk.j].clus[jk.k];}

  constexpr auto & asoa() { return asoa_;}
  constexpr auto const & asoa() const { return asoa_;}

private:
  ASOA asoa_;
};

#endif
