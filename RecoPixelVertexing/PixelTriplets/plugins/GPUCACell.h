//
// Author: Felice Pantaleo, CERN
//
#ifndef RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h
#define RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CircleEq.h"


#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"

// #define ALL_TRIPLETS

class GPUCACell {
public:

  using ptrAsInt = unsigned long long;

  static constexpr int maxCellsPerHit = CAConstants::maxCellsPerHit();
  using OuterHitOfCell = CAConstants::OuterHitOfCell;
  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  using Hits = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
  using hindex_type = siPixelRecHitsHeterogeneousProduct::hindex_type;

  using TmpTuple = GPU::VecArray<uint32_t,6>;

  using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;

  using TupleMultiplicity = CAConstants::TupleMultiplicity;

  GPUCACell() = default;
#ifdef __CUDACC__

  __device__ __forceinline__
  void init(CellNeighborsVector & cellNeighbors, CellTracksVector & cellTracks,
      Hits const & hh,
      int layerPairId, int doubletId,  
      hindex_type innerHitId, hindex_type outerHitId)
  {
    theInnerHitId = innerHitId;
    theOuterHitId = outerHitId;
    theDoubletId = doubletId;
    theLayerPairId = layerPairId;

    theInnerZ = __ldg(hh.zg_d+innerHitId);
    theInnerR = __ldg(hh.rg_d+innerHitId);

#ifdef USE_SMART_CACHE
    // link to default empty
    theOuterNeighbors = &cellNeighbors[0];
    theTracks = &cellTracks[0];
    assert(outerNeighbors().empty());
    assert(tracks().empty());
#else
    outerNeighbors().reset();
    tracks().reset();
#endif
    assert(outerNeighbors().empty());
    assert(tracks().empty());

  }


 __device__ __forceinline__
  int addOuterNeighbor(CellNeighbors::value_t t, CellNeighborsVector & cellNeighbors) {
#ifdef USE_SMART_CACHE
     if (outerNeighbors().empty()) {
       auto i = cellNeighbors.extend(); // maybe waisted....
       if (i>0) {
         auto zero = (ptrAsInt)(&cellNeighbors[0]);
         atomicCAS((ptrAsInt*)(&theOuterNeighbors),zero,(ptrAsInt)(&cellNeighbors[i]));// if fails we cannot give "i" back...
         cellNeighbors[i].reset();
       } else return -1;
     } 
#endif
     return outerNeighbors().push_back(t);  
  }

  __device__ __forceinline__
  int addTrack(CellTracks::value_t t, CellTracksVector & cellTracks) {
#ifdef USE_SMART_CACHE
     if (tracks().empty()) {
       auto i = cellTracks.extend(); // maybe waisted....
       if (i>0) {
         auto zero = (ptrAsInt)(&cellTracks[0]);
         atomicCAS((ptrAsInt*)(&theTracks),zero,(ptrAsInt)(&cellTracks[i]));
         cellTracks[i].reset();
       }
       else return -1;
     }
#endif
     return tracks().push_back(t);
  } 

#ifdef USE_SMART_CACHE
  __device__ __forceinline__ CellTracks & tracks() { return *theTracks;}
  __device__ __forceinline__ CellTracks const & tracks() const { return *theTracks;}
  __device__ __forceinline__ CellNeighbors & outerNeighbors() { return *theOuterNeighbors;}
  __device__ __forceinline__ CellNeighbors const & outerNeighbors() const { return *theOuterNeighbors;}
#else
  __device__ __forceinline__ CellTracks & tracks() { return theTracks;}
  __device__ __forceinline__ CellTracks const & tracks() const { return theTracks;}
  __device__ __forceinline__ CellNeighbors & outerNeighbors() { return theOuterNeighbors;}
  __device__ __forceinline__ CellNeighbors const & outerNeighbors() const { return theOuterNeighbors;}
#endif

  __device__ __forceinline__ float get_inner_x(Hits const & hh) const { return __ldg(hh.xg_d+theInnerHitId); }
  __device__ __forceinline__ float get_outer_x(Hits const & hh) const { return __ldg(hh.xg_d+theOuterHitId); }
  __device__ __forceinline__ float get_inner_y(Hits const & hh) const { return __ldg(hh.yg_d+theInnerHitId); }
  __device__ __forceinline__ float get_outer_y(Hits const & hh) const { return __ldg(hh.yg_d+theOuterHitId); }
  __device__ __forceinline__ float get_inner_z(Hits const & hh) const { return theInnerZ; } // { return __ldg(hh.zg_d+theInnerHitId); } // { return theInnerZ; }
  __device__ __forceinline__ float get_outer_z(Hits const & hh) const { return __ldg(hh.zg_d+theOuterHitId); }
  __device__ __forceinline__ float get_inner_r(Hits const & hh) const { return theInnerR; } // { return __ldg(hh.rg_d+theInnerHitId); } // { return theInnerR; }
  __device__ __forceinline__ float get_outer_r(Hits const & hh) const { return __ldg(hh.rg_d+theOuterHitId); }

  __device__ __forceinline__ float get_inner_detId(Hits const & hh) const { return __ldg(hh.detInd_d+theInnerHitId); }
  __device__ __forceinline__ float get_outer_detId(Hits const & hh) const { return __ldg(hh.detInd_d+theOuterHitId); }

  constexpr unsigned int get_inner_hit_id() const {
    return theInnerHitId;
  }
  constexpr unsigned int get_outer_hit_id() const {
    return theOuterHitId;
  }


  __device__
  void print_cell() const {
    printf("printing cell: %d, on layerPair: %d, innerHitId: %d, outerHitId: "
           "%d, innerradius %f, outerRadius %f \n",
           theDoubletId, theLayerPairId, theInnerHitId, theOuterHitId
    );
  }


  __device__
  bool check_alignment(Hits const & hh,
      GPUCACell const & otherCell, 
      const float ptmin,
      const float hardCurvCut) const
  {
    auto ri = get_inner_r(hh);
    auto zi = get_inner_z(hh);

    auto ro = get_outer_r(hh);
    auto zo = get_outer_z(hh);

    auto r1 = otherCell.get_inner_r(hh);
    auto z1 = otherCell.get_inner_z(hh);
    auto isBarrel = otherCell.get_outer_detId(hh)<1184;
    bool aligned = areAlignedRZ(r1, z1, ri, zi, ro, zo, ptmin, isBarrel ? 0.002f : 0.003f); // 2.f*thetaCut); // FIXME tune cuts
    return (aligned &&  dcaCut(hh, otherCell, otherCell.get_inner_detId(hh)<96 ? 0.15f : 0.25f, hardCurvCut));  // FIXME tune cuts
                            // region_origin_radius_plus_tolerance,  hardCurvCut));
  }

  __device__ __forceinline__
  static bool areAlignedRZ(float r1, float z1, float ri, float zi, float ro, float zo,
                                        const float ptmin,
                                        const float thetaCut) {
    float radius_diff = std::abs(r1 - ro);
    float distance_13_squared =
        radius_diff * radius_diff + (z1 - zo) * (z1 - zo);

    float pMin =
        ptmin * std::sqrt(distance_13_squared); // this needs to be divided by
                                                // radius_diff later

    float tan_12_13_half_mul_distance_13_squared =
        fabs(z1 * (ri - ro) + zi * (ro - r1) + zo * (r1 - ri));
    return tan_12_13_half_mul_distance_13_squared * pMin <= thetaCut * distance_13_squared * radius_diff;
  }

  
  __device__
  inline bool
  dcaCut(Hits const & hh, GPUCACell const & otherCell,
                       const float region_origin_radius_plus_tolerance,
                       const float maxCurv) const {

    auto x1 = otherCell.get_inner_x(hh);
    auto y1 = otherCell.get_inner_y(hh);

    auto x2 = get_inner_x(hh);
    auto y2 = get_inner_y(hh);

    auto x3 = get_outer_x(hh);
    auto y3 = get_outer_y(hh);

    CircleEq<float> eq(x1,y1,x2,y2,x3,y3);  

    if (eq.curvature() > maxCurv) return false;

    return std::abs(eq.dca0()) < region_origin_radius_plus_tolerance*std::abs(eq.curvature());

  }

  __device__
  inline bool 
  hole(Hits const & hh, GPUCACell const & innerCell) const {
    constexpr float r4 = 16.f;
    auto ri = innerCell.get_inner_r(hh);
    auto zi = innerCell.get_inner_z(hh);
    auto ro = get_outer_r(hh);
    auto zo = get_outer_z(hh);
    auto z4 = std::abs(zi + (r4-ri)*(zo-zi)/(ro-ri));
    return z4>25.f && z4<33.f;
  }


  // trying to free the track building process from hardcoded layers, leaving
  // the visit of the graph based on the neighborhood connections between cells.

  template<typename CM>
  __device__
  inline void find_ntuplets(
      Hits const & hh,
      GPUCACell * __restrict__ cells,
      CellTracksVector & cellTracks,
      TuplesOnGPU::Container & foundNtuplets, 
      AtomicPairCounter & apc,
      CM & tupleMultiplicity,
      TmpTuple & tmpNtuplet,
      const unsigned int minHitsPerNtuplet) const
  {
    // the building process for a track ends if:
    // it has no right neighbor
    // it has no compatible neighbor
    // the ntuplets is then saved if the number of hits it contains is greater
    // than a threshold

    tmpNtuplet.push_back_unsafe(theDoubletId);
    assert(tmpNtuplet.size()<=4);

    if(outerNeighbors().size()>0) {
      for (int j = 0; j < outerNeighbors().size(); ++j) {
        auto otherCell = outerNeighbors()[j];
        cells[otherCell].find_ntuplets(hh, cells, cellTracks, foundNtuplets, apc, tupleMultiplicity, 
                                       tmpNtuplet, minHitsPerNtuplet);
      }
    } else {  // if long enough save...
      if ((unsigned int)(tmpNtuplet.size()) >= minHitsPerNtuplet-1) {
#ifndef ALL_TRIPLETS
        // triplets accepted only pointing to the hole
        if (tmpNtuplet.size()>=3 || hole(hh, cells[tmpNtuplet[0]]))
#endif
       {
          hindex_type hits[6]; auto nh=0U;
          for (auto c : tmpNtuplet) hits[nh++] = cells[c].theInnerHitId;
          hits[nh] = theOuterHitId; 
          auto it = foundNtuplets.bulkFill(apc,hits,tmpNtuplet.size()+1);
          if (it>=0)  { // if negative is overflow....
            for (auto c : tmpNtuplet) cells[c].addTrack(it,cellTracks);
            tupleMultiplicity.countDirect(tmpNtuplet.size()+1);
          }
        }
      }
    }
    tmpNtuplet.pop_back();
    assert(tmpNtuplet.size() < 4);
  }

#endif // __CUDACC__

private:
#ifdef USE_SMART_CACHE
  CellNeighbors * theOuterNeighbors;
  CellTracks * theTracks;
#else
  CellNeighbors theOuterNeighbors;
  CellTracks theTracks;
#endif

public:
  int32_t theDoubletId;
  int32_t theLayerPairId;

private:
  float theInnerZ;
  float theInnerR;
  hindex_type theInnerHitId;
  hindex_type theOuterHitId;
};

#endif // RecoPixelVertexing_PixelTriplets_plugins_GPUCACell_h
