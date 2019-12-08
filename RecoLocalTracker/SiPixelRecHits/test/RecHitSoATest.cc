#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memsetAsync.h"

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"

#include "RecHitSoATest.h"

/*
TrackingRecHit2DCUDA and TrackingRecHit2DCPU are NOT the same type (for tracks and vertices are the same type)
they are templated with the (GPU/CPU)Traits so in principle the whole Analyzer can be (partially) templated as well
they return the same view type though (so the real analyzer algo in the header file above does not need to be templated)
*/


class RecHitSoATest : public edm::global::EDAnalyzer<> {
public:

  explicit RecHitSoATest(const edm::ParameterSet& iConfig);
  ~RecHitSoATest() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const override;
  const bool m_onGPU;
  edm::EDGetTokenT<CUDAProduct<TrackingRecHit2DCUDA>> tGpuHits;
  edm::EDGetTokenT<TrackingRecHit2DCPU> tCpuHits;
};

RecHitSoATest::RecHitSoATest(const edm::ParameterSet& iConfig) : m_onGPU(iConfig.getParameter<bool>("onGPU")) {
  if (m_onGPU) {
      tGpuHits = 
          consumes<CUDAProduct<TrackingRecHit2DCUDA>>(iConfig.getParameter<edm::InputTag>("heterogeneousPixelRecHitSrc"));
  } else {
      tCpuHits =
          consumes<TrackingRecHit2DCPU>(iConfig.getParameter<edm::InputTag>("heterogeneousPixelRecHitSrc"));
  }
}


/*
  most of (all?) the code below is boiler-plate. we hope in future to fully wrap it so that a single (templated?) source will work for both gpu and cpu 
*/
void RecHitSoATest::analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const {
  if (m_onGPU) {
    auto const & gh = iEvent.get(tGpuHits);
    CUDAScopedContextProduce ctx{gh};

    auto const & gHits = ctx.get(gh);
    auto nhits = gHits.nHits();

    if (0 == nhits) return;

    auto ws_d = GPUTraits::make_unique<double[]>(10,ctx.stream());
    cudautils::memsetAsync(ws_d, 0, 10, ctx.stream());
    // in future this bit should go in the .cu as well (otherwise it will launch the cpu version as below!)
    // in any case we cannot launch on gpu and cpu from the same file (at least with current "technology")
    int threadsPerBlock = 256;
    int blocks = (nhits + threadsPerBlock - 1) / threadsPerBlock;
    cudautils::launch(analyzeHits,{threadsPerBlock,blocks,0,ctx.stream()} , gHits.view(), nhits, ws_d.get());

  } else {
    // MIND most of the (wrapper) types are different w/r/t above!
    auto const & gHits = iEvent.get(tCpuHits);
    auto nhits = gHits.nHits();

    if (0 == nhits) return;

    auto ws_d = CPUTraits::make_unique<double[]>(10,cudaStreamDefault);
    ::memset(ws_d.get(), 0, 10*sizeof(double));  // in future cudautils::memsetAsync will work as well
    // int threadsPerBlock = 256;
    // int blocks = (nhits + threadsPerBlock - 1) / threadsPerBlock;
    //    cudautils::launch(analyzeHits,{threadsPerBlock,blocks,0,cudaStreamDefault} , gHits.view(), nhits, ws_d.get());
    analyzeHits(gHits.view(), nhits, ws_d.get());
  }
}

void RecHitSoATest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("onGPU", true);
  desc.add<edm::InputTag>("heterogeneousPixelRecHitSrc", edm::InputTag("siPixelRecHitsCUDAPreSplitting"));
  descriptions.add("RecHitSoATest", desc);
}

DEFINE_FWK_MODULE(RecHitSoATest);
