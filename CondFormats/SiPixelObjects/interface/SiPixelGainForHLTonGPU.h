#ifndef CondFormats_SiPixelObjects_interface_SiPixelGainForHLTonGPU_h
#define CondFormats_SiPixelObjects_interface_SiPixelGainForHLTonGPU_h

#include <cstdint>
#include <cstdio>
#include <tuple>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

struct SiPixelGainForHLTonGPU_DecodingStructure {
  uint8_t gain;
  uint8_t ped;
};

// copy of SiPixelGainCalibrationForHLT
class SiPixelGainForHLTonGPU {
public:
  using DecodingStructure = SiPixelGainForHLTonGPU_DecodingStructure;

  using Range = std::pair<uint32_t, uint32_t>;

  inline __device__ std::pair<float, float> getPedAndGain(
      uint32_t moduleInd, int col, int row, bool& isDeadColumn, bool& isNoisyColumn) const {
    constexpr uint32_t numberOfRowsAveragedOver = 80;
    constexpr uint32_t deadFlag = 255;
    constexpr uint32_t noisyFlag = 254;
    constexpr uint32_t lengthOfAveragedDataInEachColumn =
        2;  // we always only have two values per column averaged block

    auto range = rangeAndCols_[moduleInd].first;
    auto nCols = rangeAndCols_[moduleInd].second;

    // determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
    uint32_t lengthOfColumnData = (range.second - range.first) / nCols;
    uint32_t numberOfDataBlocksToSkip = row / numberOfRowsAveragedOver;

    auto offset = range.first + col * lengthOfColumnData + lengthOfAveragedDataInEachColumn * numberOfDataBlocksToSkip;

    assert(offset < range.second);
    assert(offset < 3088384);
    assert(0 == offset % 2);

    // type punning
    union U {
      DecodingStructure ds;
      uint16_t u16;
    };
    U u;
    u.u16 = __ldg((uint16_t const*)(v_pedestals_ + offset / 2));
    auto s = u.ds;

    isDeadColumn = (s.ped & 0xFF) == deadFlag;
    isNoisyColumn = (s.ped & 0xFF) == noisyFlag;

    return std::make_pair(decodePed(s.ped & 0xFF), decodeGain(s.gain & 0xFF));
  }

  constexpr float decodeGain(unsigned int gain) const { return gain * gainPrecision_ + minGain_; }
  constexpr float decodePed(unsigned int ped) const { return ped * pedPrecision_ + minPed_; }

  DecodingStructure* v_pedestals_;
  std::pair<Range, int> rangeAndCols_[gpuClustering::maxNumModules];

  float minPed_, minGain_;
  float pedPrecision_, gainPrecision_;
};

#endif  // CondFormats_SiPixelObjects_interface_SiPixelGainForHLTonGPU_h
