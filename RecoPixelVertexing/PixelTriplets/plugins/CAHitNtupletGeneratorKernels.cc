#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernelsImpl.h"

template<>
void CAHitNtupletGeneratorKernelsCPU::printCounters(Counters const * counters) {
   kernel_printCounters(counters);
}


template<>
void CAHitNtupletGeneratorKernelsCPU::fillHitDetIndices(HitsView const * hv, TkSoA * tracks_d, cudaStream_t) {
  kernel_fillHitDetIndices(&tracks_d->hitIndices, hv, &tracks_d->detIndices);
}



template<>
void CAHitNtupletGeneratorKernelsCPU::buildDoublets(HitsOnCPU const &hh, cuda::stream_t<> &stream) {
  auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

  // in principle we can use "nhits" to heuristically dimension the workspace...
  // overkill to use template here (std::make_unique would suffice)
  edm::Service<CUDAService> cs;
  device_isOuterHitOfCell_ = Traits:: template make_unique<GPUCACell::OuterHitOfCell[]>(cs, std::max(1U,nhits), stream);
  assert(device_isOuterHitOfCell_.get());
  gpuPixelDoublets::initDoublets(device_isOuterHitOfCell_.get(),
                                                                                   nhits,
                                                                                   device_theCellNeighbors_,
                                                                                   device_theCellNeighborsContainer_.get(),
                                                                                   device_theCellTracks_,
                                                                                   device_theCellTracksContainer_.get());

  device_theCells_ = Traits:: template make_unique<GPUCACell[]>(cs, CAConstants::maxNumberOfDoublets(), stream);

  if (0 == nhits)
    return;  // protect against empty events

  // FIXME avoid magic numbers
  auto nActualPairs=gpuPixelDoublets::nPairs;
  if (!m_params.includeJumpingForwardDoublets_) nActualPairs = 15;
  if (m_params.minHitsPerNtuplet_>3) {
    nActualPairs = 13;
  }

  assert(nActualPairs<=gpuPixelDoublets::nPairs);
  gpuPixelDoublets::getDoubletsFromHisto(device_theCells_.get(),
                                                                         device_nCells_,
                                                                         device_theCellNeighbors_,
                                                                         device_theCellTracks_,
                                                                         hh.view(),
                                                                         device_isOuterHitOfCell_.get(),
                                                                         nActualPairs,
                                                                         m_params.idealConditions_,
                                                                         m_params.doClusterCut_,
                                                                         m_params.doZCut_,
                                                                         m_params.doPhiCut_);


}


template<>
void CAHitNtupletGeneratorKernelsCPU::launchKernels(
    HitsOnCPU const &hh,
    TkSoA * tracks_d,
    cudaStream_t cudaStream) {

  auto * tuples_d = &tracks_d->hitIndices;
  auto * quality_d = (Quality*)(&tracks_d->m_quality);

  assert(tuples_d && quality_d);

  auto nhits = hh.nHits();
  assert(nhits <= pixelGPUConstants::maxNumberOfHits);

  // std::cout << "N hits " << nhits << std::endl;
  // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

  //
  // applying conbinatoric cleaning such as fishbone at this stage is too expensive
  //

  kernel_connect(
      device_hitTuple_apc_,
      device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
      hh.view(),
      device_theCells_.get(),
      device_nCells_,
      device_theCellNeighbors_,
      device_isOuterHitOfCell_.get(),
      m_params.hardCurvCut_,
      m_params.ptmin_,
      m_params.CAThetaCutBarrel_,
      m_params.CAThetaCutForward_,
      m_params.dcaCutInnerTriplet_,
      m_params.dcaCutOuterTriplet_);


 if (nhits > 1 && m_params.earlyFishbone_) {
    fishbone(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, false);
  }


  int nIter = m_params.doIterations_ ? 3 : 1;
  if (m_params.minHitsPerNtuplet_>3) nIter=1;
  for (int startLayer=0; startLayer<nIter; ++startLayer) {
    kernel_find_ntuplets(hh.view(),
                                                                     device_theCells_.get(),
                                                                     device_nCells_,
                                                                     device_theCellTracks_,
                                                                     tuples_d,
                                                                     device_hitTuple_apc_,
                                                                     quality_d,
                                                                     m_params.minHitsPerNtuplet_,
                                                                     m_params.doIterations_ ? startLayer : -1);
    if (m_params.doIterations_ || m_params.doStats_)
    kernel_mark_used(hh.view(),
                                                                   device_theCells_.get(),
                                                                   device_nCells_);
  }


  cudautils::finalizeBulk(device_hitTuple_apc_, tuples_d);

  // remove duplicates (tracks that share a doublet)
  kernel_earlyDuplicateRemover(
      device_theCells_.get(), device_nCells_, tuples_d, quality_d);



 if (m_params.doStats_) {
    kernel_checkOverflows(tuples_d,
                                                                        device_tupleMultiplicity_.get(),
                                                                        device_hitTuple_apc_,
                                                                        device_theCells_.get(),
                                                                        device_nCells_,
                                                                        device_theCellNeighbors_,
                                                                        device_theCellTracks_,
                                                                        device_isOuterHitOfCell_.get(),
                                                                        nhits,
                                                                        counters_);
  }

}
