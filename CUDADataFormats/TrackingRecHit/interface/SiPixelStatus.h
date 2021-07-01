#ifndef CUDADataFormats_TrackingRecHit_interface_SiPixelStatus_H
#define CUDADataFormats_TrackingRecHit_interface_SiPixelStatus_H

#include <cstdint>

// more information on bit fields : https://en.cppreference.com/w/cpp/language/bit_field
struct SiPixelStatus {
  uint8_t isBigX : 1;
  uint8_t isOneX : 1;
  uint8_t isBigY : 1;
  uint8_t isOneY : 1;
  uint8_t qBin : 3;
};

#endif
