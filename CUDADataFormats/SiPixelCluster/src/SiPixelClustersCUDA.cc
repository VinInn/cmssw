#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t maxClusters, cuda::stream_t<>& stream) {
  moduleStart_d     = cudautils::make_device_unique<uint32_t[]>(maxClusters+1, stream);
  clusInModule_d    = cudautils::make_device_unique<uint32_t[]>(maxClusters, stream);
  moduleId_d        = cudautils::make_device_unique<uint32_t[]>(maxClusters, stream);
  clusModuleStart_d = cudautils::make_device_unique<uint32_t[]>(maxClusters+1, stream);

  auto view = cudautils::make_host_unique<DeviceConstView>(stream);
  view->moduleStart_ = moduleStart_d.get();
  view->clusInModule_ = clusInModule_d.get();
  view->moduleId_ = moduleId_d.get();
  view->clusModuleStart_ = clusModuleStart_d.get();

  view_d = cudautils::make_device_unique<DeviceConstView>(stream);
  cudautils::copyAsync(view_d, view, stream);
}
