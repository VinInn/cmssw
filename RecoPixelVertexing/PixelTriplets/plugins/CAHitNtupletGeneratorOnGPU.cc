//
// Original Author: Felice Pantaleo, CERN
//

#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "CAHitNtupletGeneratorOnGPU.h"

namespace {

  template <typename T>
  T sqr(T x) {
    return x * x;
  }

  CAHitNtupletGeneratorKernels::QualityCuts makeQualityCuts(edm::ParameterSet const& pset) {
    auto coeff = pset.getParameter<std::vector<double>>("chi2Coeff");
    if (coeff.size() != 4) {
      throw edm::Exception(edm::errors::Configuration, "CAHitNtupletGeneratorOnGPU.trackQualityCuts.chi2Coeff must have 4 elements");
    }
    return CAHitNtupletGeneratorKernels::QualityCuts {
      // polynomial coefficients for the pT-dependent chi2 cut
      { (float) coeff[0], (float) coeff[1], (float) coeff[2], (float) coeff[3] },
      // max pT used to determine the chi2 cut
      (float) pset.getParameter<double>("chi2MaxPt"),
      // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
      (float) pset.getParameter<double>("chi2Scale"),
      // regional cuts for triplets
      {
        (float) pset.getParameter<double>("tripletMaxTip"),
        (float) pset.getParameter<double>("tripletMinPt"),
        (float) pset.getParameter<double>("tripletMaxZip")
      },
      // regional cuts for quadruplets
      {
        (float) pset.getParameter<double>("quadrupletMaxTip"),
        (float) pset.getParameter<double>("quadrupletMinPt"),
        (float) pset.getParameter<double>("quadrupletMaxZip")
      }
    };
  }

}  // namespace

using namespace std;

CAHitNtupletGeneratorOnGPU::CAHitNtupletGeneratorOnGPU(const edm::ParameterSet &cfg, edm::ConsumesCollector &iC)
    : m_params(cfg.getParameter<unsigned int>("minHitsPerNtuplet"),
              cfg.getParameter<bool>("useRiemannFit"),
              cfg.getParameter<bool>("fit5as4"),
              cfg.getParameter<bool>("includeJumpingForwardDoublets"),
              cfg.getParameter<bool>("earlyFishbone"),
              cfg.getParameter<bool>("lateFishbone"),
              cfg.getParameter<bool>("idealConditions"),
              cfg.getParameter<bool>("fillStatistics"),
              cfg.getParameter<bool>("doClusterCut"),
              cfg.getParameter<bool>("doZCut"),
              cfg.getParameter<bool>("doPhiCut"),
              cfg.getParameter<bool>("doIterations"),
              cfg.getParameter<double>("ptmin"),
              cfg.getParameter<double>("CAThetaCutBarrel"),
              cfg.getParameter<double>("CAThetaCutForward"),
              cfg.getParameter<double>("hardCurvCut"),
              cfg.getParameter<double>("dcaCutInnerTriplet"),
              cfg.getParameter<double>("dcaCutOuterTriplet"),
              makeQualityCuts(cfg.getParameterSet("trackQualityCuts"))) {

#ifdef    DUMP_GPU_TK_TUPLES
      printf("TK: %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
             "tid", "qual", "nh","phi","tip","pt","cot","zip","eta","chil","chic",
             "h1","h2","h3","h4","h5");
#endif

  cudaCheck(cudaMalloc(&m_counters, sizeof(Counters)));
  cudaCheck(cudaMemset(m_counters, 0, sizeof(Counters)));

}

CAHitNtupletGeneratorOnGPU::~CAHitNtupletGeneratorOnGPU(){
 if (m_params.doStats_) {
    // crash on multi-gpu processes
    CAHitNtupletGeneratorKernels::printCounters(m_counters);
  }
  cudaFree(m_counters);
}


void CAHitNtupletGeneratorOnGPU::fillDescriptions(edm::ParameterSetDescription &desc) {
  // 87 cm/GeV = 1/(3.8T * 0.3)
  // take less than radius given by the hardPtCut and reject everything below
  // auto hardCurvCut = 1.f/(0.35 * 87.f);
  desc.add<double>("ptmin", 0.9f)->setComment("Cut on minimum pt");
  desc.add<double>("CAThetaCutBarrel", 0.002f)->setComment("Cut on RZ alignement for Barrel");
  desc.add<double>("CAThetaCutForward", 0.003f)->setComment("Cut on RZ alignment for Forward");
  desc.add<double>("hardCurvCut", 1.f / (0.35 * 87.f))->setComment("Cut on minimum curvature");
  desc.add<double>("dcaCutInnerTriplet", 0.15f)->setComment("Cut on origin radius when the inner hit is on BPix1");
  desc.add<double>("dcaCutOuterTriplet", 0.25f)->setComment("Cut on origin radius when the outer hit is on BPix1");
  desc.add<bool>("earlyFishbone", true);
  desc.add<bool>("lateFishbone", false);
  desc.add<bool>("idealConditions", true);
  desc.add<bool>("fillStatistics", false);
  desc.add<unsigned int>("minHitsPerNtuplet", 4);
  desc.add<bool>("includeJumpingForwardDoublets", false);
  desc.add<bool>("fit5as4", true);
  desc.add<bool>("doClusterCut", true);
  desc.add<bool>("doZCut", true);
  desc.add<bool>("doPhiCut", true);
  desc.add<bool>("doIterations", false);
  desc.add<bool>("useRiemannFit", false)->setComment("true for Riemann, false for BrokenLine");

  edm::ParameterSetDescription trackQualityCuts;
  trackQualityCuts.add<double>("chi2MaxPt", 10.)->setComment("max pT used to determine the pT-dependent chi2 cut");
  trackQualityCuts.add<std::vector<double>>("chi2Coeff", { 0.68177776, 0.74609577, -0.08035491, 0.00315399 })
    ->setComment("Polynomial coefficients to derive the pT-dependent chi2 cut");
  trackQualityCuts.add<double>("chi2Scale", 30.)->setComment("Factor to multiply the pT-dependent chi2 cut (currently: 30 for the broken line fit, 45 for the Riemann fit)");
  trackQualityCuts.add<double>("tripletMinPt", 0.5)->setComment("Min pT for triplets, in GeV");
  trackQualityCuts.add<double>("tripletMaxTip", 0.3)->setComment("Max |Tip| for triplets, in cm");
  trackQualityCuts.add<double>("tripletMaxZip", 12.)->setComment("Max |Zip| for triplets, in cm");
  trackQualityCuts.add<double>("quadrupletMinPt", 0.3)->setComment("Min pT for quadruplets, in GeV");
  trackQualityCuts.add<double>("quadrupletMaxTip", 0.5)->setComment("Max |Tip| for quadruplets, in cm");
  trackQualityCuts.add<double>("quadrupletMaxZip", 12.)->setComment("Max |Zip| for quadruplets, in cm");
  desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
    ->setComment("Quality cuts based on the results of the track fit:\n  - apply a pT-dependent chi2 cut;\n  - apply \"region cuts\" based on the fit results (pT, Tip, Zip).");
}


PixelTrackCUDA CAHitNtupletGeneratorOnGPU::makeTuplesAsync(TrackingRecHit2DCUDA const& hits_d,
                                                           float bfield,
                                                           cuda::stream_t<>& stream) const {

  PixelTrackCUDA tracks(hits_d.view(),stream);

  auto * soa = tracks.soa();
  
  CAHitNtupletGeneratorKernels kernels(m_params);
  kernels.counters_ = m_counters;
  HelixFitOnGPU fitter(bfield,m_params.fit5as4_);

  kernels.allocateOnGPU(stream);
  fitter.allocateOnGPU(&(soa->hitIndices), kernels.tupleMultiplicity(), soa);

  kernels.buildDoublets(hits_d, stream);
  kernels.launchKernels(hits_d, soa, stream.id());
  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernels(hits_d, hits_d.nHits(), CAConstants::maxNumberOfQuadruplets(), stream);
  } else {
    fitter.launchBrokenLineKernels(hits_d, hits_d.nHits(), CAConstants::maxNumberOfQuadruplets(), stream);
  }
  kernels.classifyTuples(hits_d, soa, stream.id());

  return tracks;
}

