#define GPU_DEBUG
#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernelsImpl.h"

template <>
void CAHitNtupletGeneratorKernelsGPU::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d, cudaStream_t cudaStream) {
  auto blockSize = 128;
  auto numberOfBlocks = (HitContainer::capacity() + blockSize - 1) / blockSize;

  kernel_fillHitDetIndices<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      &tracks_d->hitIndices, hv, &tracks_d->detIndices);
  cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, cudaStream_t cudaStream) {
  // these are pointer on GPU!
  auto *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  // zero tuples
  cms::cuda::launchZero(tuples_d, cudaStream);

  auto nhits = hh.nHits();
  assert(nhits <= pixelGPUConstants::maxNumberOfHits);

  // std::cout << "N hits " << nhits << std::endl;
  // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

  //
  // applying conbinatoric cleaning such as fishbone at this stage is too expensive
  //

  auto nthTot = 64;
  auto stride = 4;
  int blockSize = nthTot / stride;
  int numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  auto rescale = numberOfBlocks / 65536;
  blockSize *= (rescale + 1);
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  assert(numberOfBlocks < 65536);
  assert(blockSize > 0 && 0 == blockSize % 16);
  dim3 blks(1, numberOfBlocks, 1);
  dim3 thrs(stride, blockSize, 1);

  kernel_connect<<<blks, thrs, 0, cudaStream>>>(
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
  cudaCheck(cudaGetLastError());

  if (nhits > 1 && m_params.earlyFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    gpuPixelDoublets::fishbone<<<blks, thrs, 0, cudaStream>>>(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, false);
    cudaCheck(cudaGetLastError());
  }

  blockSize = 64;
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  kernel_find_ntuplets<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                                     device_theCells_.get(),
                                                                     device_nCells_,
                                                                     device_theCellTracks_,
                                                                     tuples_d,
                                                                     device_hitTuple_apc_,
                                                                     quality_d,
                                                                     m_params.minHitsPerNtuplet_);
  cudaCheck(cudaGetLastError());

  if (m_params.doStats_)
    kernel_mark_used<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(), device_theCells_.get(), device_nCells_);
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  blockSize = 128;
  numberOfBlocks = (HitContainer::totbins() + blockSize - 1) / blockSize;
  cms::cuda::finalizeBulk<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitTuple_apc_, tuples_d);

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  kernel_earlyDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      device_theCells_.get(), device_nCells_, tuples_d, quality_d);
  cudaCheck(cudaGetLastError());

  // fill multiplicity histos 
  static cms::cuda::CoopKernelConfig config(CAConstants::TupleMultiplicity::nthreads());
  auto kc = config.getConfig(kernel_fillMultiplicity);
  blockSize = kc.second;
  numberOfBlocks = int(3 * CAConstants::maxTuples() / 4 + blockSize - 1) / blockSize;
  numberOfBlocks = std::min(numberOfBlocks,kc.first);
  launch_cooperative(kernel_fillMultiplicity,{numberOfBlocks, blockSize, 0, cudaStream},
      tuples_d, quality_d, device_tupleMultiplicity_.get());
  cudaCheck(cudaGetLastError());

  if (nhits > 1 && m_params.lateFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    gpuPixelDoublets::fishbone<<<blks, thrs, 0, cudaStream>>>(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, true);
    cudaCheck(cudaGetLastError());
  }

  if (m_params.doStats_) {
    numberOfBlocks = (std::max(nhits, m_params.maxNumberOfDoublets_) + blockSize - 1) / blockSize;
    kernel_checkOverflows<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d,
                                                                        device_tupleMultiplicity_.get(),
                                                                        device_hitTuple_apc_,
                                                                        device_theCells_.get(),
                                                                        device_nCells_,
                                                                        device_theCellNeighbors_,
                                                                        device_theCellTracks_,
                                                                        device_isOuterHitOfCell_.get(),
                                                                        nhits,
                                                                        m_params.maxNumberOfDoublets_,
                                                                        counters_);
    cudaCheck(cudaGetLastError());
  }
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}


template <>
void CAHitNtupletGeneratorKernelsGPU::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d, cudaStream_t cudaStream) {
  // these are pointer on GPU!
  auto const *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  auto blockSize = 64;

  // classify tracks based on kinematics
  int numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
  kernel_classifyTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, tracks_d, m_params.cuts_, quality_d);
  cudaCheck(cudaGetLastError());

  if (m_params.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    kernel_fishboneCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        device_theCells_.get(), device_nCells_, quality_d);
    cudaCheck(cudaGetLastError());
  }

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  kernel_fastDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      device_theCells_.get(), device_nCells_, tuples_d, tracks_d);
  cudaCheck(cudaGetLastError());

  if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
    // fill hit->track "map"
     static cms::cuda::CoopKernelConfig config(CAConstants::HitToTuple::nthreads());
     auto kc = config.getConfig(kernel_fillHitInTracks);
     blockSize = kc.second;
     numberOfBlocks = int(3 * CAConstants::maxTuples() / 4 + blockSize - 1) / blockSize;
     numberOfBlocks = std::min(numberOfBlocks,kc.first);
     launch_cooperative(kernel_fillHitInTracks,{numberOfBlocks, blockSize, 0, cudaStream},
        tuples_d, quality_d, device_hitToTuple_.get());
     cudaCheck(cudaGetLastError());
  }
  if (m_params.minHitsPerNtuplet_ < 4) {
    // remove duplicates (tracks that share a hit)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    kernel_tripletCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
  }

  if (m_params.doStats_) {
    // counters (add flag???)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    kernel_doStatsForHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitToTuple_.get(), counters_);
    cudaCheck(cudaGetLastError());
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    kernel_doStatsForTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, counters_);
    cudaCheck(cudaGetLastError());
  }
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  ++iev;
  kernel_print_found_ntuplets<<<1, 32, 0, cudaStream>>>(
      hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get(), 100, iev);
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::printCounters(Counters const *counters) {
  kernel_printCounters<<<1, 1>>>(counters);
}
