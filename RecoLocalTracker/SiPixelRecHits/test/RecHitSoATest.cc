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

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"

#include "RecHitSoATest.h"

class RecHitSoATest : public edm::global::EDAnalyzer<> {
public:

  explicit RecHitSoATest(const edm::ParameterSet& iConfig);
  ~RecHitSoATest() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const override;
  const bool m_onGPU;
  edm::EDGetTokenT<CUDAProduct<TrackingRecHit2DCUDA>> tGpuHits;
  edm::EDGetTokenT<TrackingRecHit2DCUDA> tCpuHits;
};

RecHitSoATest::RecHitSoATest(const edm::ParameterSet& iConfig) : m_onGPU(iConfig.getParameter<bool>("onGPU")) {
  if (m_onGPU) {
      tGpuHits = 
          consumes<CUDAProduct<TrackingRecHit2DCUDA>>(iConfig.getParameter<edm::InputTag>("heterogeneousPixelRecHitSrc"));
  } else {
      tCpuHits =
          consumes<TrackingRecHit2DCUDA>(iConfig.getParameter<edm::InputTag>("heterogeneousPixelRecHitSrc"));
  }
}

void RecHitSoATest::analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const {
  if (m_onGPU) {
    auto const & gh = iEvent.get(tGpuHits);
    CUDAScopedContextProduce ctx{gh};

    auto const & gHits = ctx.get(gh);
    auto nhits = gHits.nHits();

    if (0 == nhits) return;

    auto ws_d = GPUTraits::make_unique<double[]>(10,ctx.stream());
    cudaCheck(cudaMemsetAsync(ws_d.get(), 0, 10*sizeof(double), ctx.stream()));

    int threadsPerBlock = 256;
    int blocks = (nhits + threadsPerBlock - 1) / threadsPerBlock;
    cudautils::launch(analyzeHits,{threadsPerBlock,blocks,0,ctx.stream()} , gHits.view(), nhits, ws_d.get());

  } else {
    
    auto const & gHits = iEvent.get(tCpuHits);
    auto nhits = gHits.nHits();

    if (0 == nhits) return;

    auto ws_d = CPUTraits::make_unique<double[]>(10,cudaStreamDefault);
    ::memset(ws_d.get(), 0, 10*sizeof(double));
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
  descriptions.add("clusterTPCUDAdump", desc);
}

DEFINE_FWK_MODULE(RecHitSoATest);
