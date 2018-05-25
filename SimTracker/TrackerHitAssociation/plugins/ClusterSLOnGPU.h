// gpu
#include <cuda_runtime.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "EventFilter/SiPixelRawToDigi/plugins/RawToDigiGPU.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/PixelRecHits.h"

struct ClusterSLGPU {
 ClusterSLGPU(){alloc();}
 void alloc();

 ClusterSLGPU * me_d;
 std::array<uint32_t,3> * links_d;
 uint32_t * tkId_d;
 uint32_t * tkId2_d;
 uint32_t * n1_d;
 uint32_t * n2_d;

 static constexpr uint32_t MAX_DIGIS = 2000*150;
 static constexpr uint32_t MaxNumModules = 2000;

};

namespace clusterSLOnGPU {

  void wrapper(context const & dd, HitsOnGPU const & hh, ClusterSLGPU const & sl, uint32_t n);


}
