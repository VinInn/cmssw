#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoubletsAlgos_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoupletsAlgos_h

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"

#include "GPUCACell.h"
#include "CAConstants.h"

// #define DEBUG_DOUBLETS

// useful for benchmark
// #define ONLY_PHICUT
// #define NO_ZCUT
// #define NO_CLSCUT

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

namespace gpuPixelDoubletsAlgos {

  constexpr uint32_t MaxNumOfDoublets = CAConstants::maxNumberOfDoublets();  // not really relevant

  constexpr uint32_t MaxNumOfActiveDoublets = CAConstants::maxNumOfActiveDoublets();


  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  __device__
  __forceinline__
  void doubletsFromHisto(uint8_t const * __restrict__ layerPairs,
                         uint32_t nPairs,
                         GPUCACell * cells,
                         uint32_t * nCells,
                         CellNeighborsVector * cellNeighbors, CellTracksVector * cellTracks,
                         TrackingRecHit2DSOAView const &  __restrict__ hh,
                         GPUCACell::OuterHitOfCell * isOuterHitOfCell,
                         int16_t const * __restrict__ phicuts,
                         float const * __restrict__ minz,
                         float const * __restrict__ maxz,
                         float const * __restrict__ maxr,
                         bool ideal_cond)
  {

#ifndef NO_CLSCUT 
    // ysize cuts (z in the barrel)  times 8
    constexpr int minYsizeB1=36;
    constexpr int minYsizeB2=28;
    constexpr int maxDYsize12=28;
    constexpr int maxDYsize=20;
#endif


    using Hist = TrackingRecHit2DSOAView::Hist;

    auto const & __restrict__ hist = hh.phiBinner();
    uint32_t const * __restrict__ offsets = hh.hitsLayerStart();
    assert(offsets);

    auto layerSize = [=](uint8_t li) { return offsets[li+1]-offsets[li]; };

    // nPairsMax to be optimized later (originally was 64).
    // If it should be much bigger, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    const int nPairsMax = 16;
    assert(nPairs <= nPairsMax);
    __shared__ uint32_t innerLayerCumulativeSize[nPairsMax];
    __shared__ uint32_t ntot;
    if (threadIdx.y==0 && threadIdx.x==0) {
      innerLayerCumulativeSize[0] = layerSize(layerPairs[0]);
      for (uint32_t i = 1; i < nPairs; ++i) {
        innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i-1] + layerSize(layerPairs[2*i]);
      }
      ntot = innerLayerCumulativeSize[nPairs-1];
    }
    __syncthreads();

    // x runs faster
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    auto first = threadIdx.x;
    auto stride = blockDim.x;
    for (auto j = idy; j < ntot; j += blockDim.y * gridDim.y ) {

      int32_t pairLayerId=0;
      while (j >= innerLayerCumulativeSize[pairLayerId++]);
      --pairLayerId; // move to lower_bound ??

      assert(pairLayerId < nPairs);
      assert(pairLayerId >=0);
      assert(j < innerLayerCumulativeSize[pairLayerId]);
      assert(0 == pairLayerId || j >= innerLayerCumulativeSize[pairLayerId-1]);

      uint8_t inner = layerPairs[2*pairLayerId];
      uint8_t outer = layerPairs[2*pairLayerId+1];
      assert(outer > inner);
      assert(outer<10);

      auto hoff = Hist::histOff(outer);

      auto i = (0 == pairLayerId) ? j : j-innerLayerCumulativeSize[pairLayerId-1];
      i += offsets[inner];

      // printf("Hit in Layer %d %d %d %d\n", i, inner, pairLayerId, j);

      assert(i >= offsets[inner]);
      assert(i < offsets[inner+1]);

      // found hit corresponding to our cuda thread, now do the job
      auto mez = hh.zGlobal(i);

#ifndef NO_ZCUT
     if (mez<minz[pairLayerId] || mez>maxz[pairLayerId]) continue;
#endif

#ifndef NO_CLSCUT
      auto mes = hh.clusterSizeY(i);

      // if ideal treat inner ladder as outer
      auto mi = hh.detectorIndex(i);
      if (inner==0) assert(mi<96);    
      const bool isOuterLadder = ideal_cond ? true : 0 == (mi/8)%2; // only for B1/B2/B3 B4 is opposite, FPIX:noclue...


      if (inner==0 && outer>3 && isOuterLadder)  // B1 and F1
         if (mes>0 && mes<minYsizeB1) continue; // only long cluster  (5*8)
      if (inner==1 && outer>3)  // B2 and F1
         if (mes>0 && mes<minYsizeB2) continue;
#endif // NO_CLSCUT

      auto mep = hh.iphi(i);
      auto mer = hh.rGlobal(i);

      assert(mez==mez);    assert(mer==mer);

      constexpr float z0cut = 12.f;                     // cm
      constexpr float hardPtCut = 0.5f;                 // GeV
      constexpr float minRadius = hardPtCut * 87.78f;   // cm (1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field)
      constexpr float minRadius2T4 = 4.f*minRadius*minRadius;
      auto ptcut = [&](int j) {
        auto r2t4 = minRadius2T4;
        auto ri = mer;
        auto ro = hh.rGlobal(j);
        assert(ri==ri); assert(ro==ro);
        auto dphi = short2phi( min( abs(int16_t(mep-hh.iphi(j))), abs(int16_t(hh.iphi(j)-mep)) ) );
        return dphi*dphi * (r2t4 - ri*ro) > (ro-ri)*(ro-ri);
      };
      auto z0cutoff = [&](int j) {
        auto zo = hh.zGlobal(j);;
        auto ro = hh.rGlobal(j);
        assert(zo==zo);    assert(ro==ro);
        auto dr = ro-mer;
        return dr > maxr[pairLayerId] ||
          dr<0 || std::abs((mez*ro - mer*zo)) > z0cut*dr;
      };

#ifndef NO_CLSCUT
      auto zsizeCut = [&](int j) {
        auto onlyBarrel = outer<4;
        auto so = hh.clusterSizeY(j);
        auto dy = inner==0 ? ( isOuterLadder ? maxDYsize12: 100 ) : maxDYsize;
        return onlyBarrel && mes>0 && so>0 && std::abs(so-mes)>dy;
      };
#endif

      auto iphicut = phicuts[pairLayerId];

      auto kl = Hist::bin(int16_t(mep-iphicut));
      auto kh = Hist::bin(int16_t(mep+iphicut));
      auto incr = [](auto & k) { return k = (k+1) % Hist::nbins();};

#ifdef DEBUG_DOUBLETS
      int  tot  = 0;
      int  nmin = 0;
      int nk=0;
      int tooMany=0;
#endif

      auto khh = kh;
      incr(khh);
      for (auto kk = kl; kk != khh; incr(kk)) {
#ifdef DEBUG_DOUBLETS
        nk++;
        if (kk != kl && kk != kh)
          nmin += hist.size(kk+hoff);
#endif
        auto const * __restrict__ p = hist.begin(kk+hoff);
        auto const * __restrict__ e = hist.end(kk+hoff);
        p+=first;
        for (;p < e; p+=stride) {
          auto oi=__ldg(p);
          assert(oi>=offsets[outer]);
          assert(oi<offsets[outer+1]);

          if (std::min(std::abs(int16_t(hh.iphi(oi)-mep)), std::abs(int16_t(mep-hh.iphi(oi)))) > iphicut)
            continue;
#ifndef ONLY_PHICUT
#ifndef NO_CLSCUT
          if (zsizeCut(oi)) continue;
#endif
          if (z0cutoff(oi) || ptcut(oi)) continue;
#endif
          auto ind = atomicAdd(nCells, 1); 
          if (ind>=MaxNumOfDoublets) {atomicSub(nCells, 1); break; } // move to SimpleVector??
          // int layerPairId, int doubletId, int innerHitId, int outerHitId)
          cells[ind].init(*cellNeighbors, *cellTracks, hh, pairLayerId, ind, i, oi);
          isOuterHitOfCell[oi].push_back(ind);
#ifdef DEBUG_DOUBLETS
          if (isOuterHitOfCell[oi].full()) ++tooMany;
          ++tot;
#endif
        }
      }
#ifdef DEBUG_DOUBLETS
      if (tooMany > 0)
        printf("OuterHitOfCell full for %d in at %d, layer %d/%d, %d,%d,%d %d\n", i, *nCells, int(inner), int(outer), nk,nmin, tot, tooMany);
#endif
    }  // loop in block...
  }

} // namespace end

#endif // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoupletsAlgos_h
