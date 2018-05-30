#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h

#include <cstdint>
#include <vector>

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"


namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

struct context;

struct HitsOnGPU{
   HitsOnGPU * me_d;
   float * bs_d;
   uint32_t * hitsModuleStart_d;
   uint32_t * hitsLayerStart_d;
   int32_t  * charge_d;
   uint16_t * detInd_d;
   float *xg_d, *yg_d, *zg_d, *rg_d;
   float *xl_d, *yl_d;
   float *xerr_d, *yerr_d;
   int16_t * iphi_d;
   uint16_t * sortIndex_d;
   uint16_t * mr_d;

   using Hist = HistoContainer<int16_t,7,8>;
   Hist * hist_d;
};

struct HitsOnCPU {
 explicit HitsOnCPU(uint32_t nhits) :
  charge(nhits),xl(nhits),yl(nhits),xe(nhits),ye(nhits), mr(nhits){}
 uint32_t hitsModuleStart[2001];
 std::vector<int32_t> charge;
 std::vector<float> xl, yl;
 std::vector<float> xe, ye;
 std::vector<uint16_t> mr;
};


HitsOnGPU allocHitsOnGPU();

HitsOnCPU pixelRecHits_wrapper(
      context const & c,
      float const * bs,
      pixelCPEforGPU::ParamsOnGPU const * cpeParams,
      uint32_t ndigis,
      uint32_t nModules, // active modules (with digis)
      HitsOnGPU & hh
);

#endif // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
