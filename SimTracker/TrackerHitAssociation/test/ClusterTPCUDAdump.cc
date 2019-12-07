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

#include "SimTracker/TrackerHitAssociation/interface/trackerHitAssociationHeterogeneous.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"

void analyzeClusterTP(SiPixelDigisCUDA::DeviceConstView const* dd, uint32_t ndigis, TrackingRecHit2DSOAView const* hhp, uint32_t nhits, trackerHitAssociationHeterogeneous::ClusterSLView);

class ClusterTPCUDAdump : public edm::global::EDAnalyzer<> {
public:
  using ClusterSLGPU = trackerHitAssociationHeterogeneous::ClusterSLView;
  using Clus2TP = ClusterSLGPU::Clus2TP;
  using ProductCUDA = trackerHitAssociationHeterogeneous::ProductCUDA;

  explicit ClusterTPCUDAdump(const edm::ParameterSet& iConfig);
  ~ClusterTPCUDAdump() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const override;
  const bool m_onGPU;
  edm::EDGetTokenT<CUDAProduct<SiPixelDigisCUDA>> tGpuDigis;
  edm::EDGetTokenT<CUDAProduct<TrackingRecHit2DCUDA>> tGpuHits;
  edm::EDGetTokenT<CUDAProduct<ProductCUDA>> tokenGPU_;
};

ClusterTPCUDAdump::ClusterTPCUDAdump(const edm::ParameterSet& iConfig) : m_onGPU(iConfig.getParameter<bool>("onGPU")) {
  if (m_onGPU) {
    tokenGPU_ = consumes<CUDAProduct<ProductCUDA>>(iConfig.getParameter<edm::InputTag>("clusterTP"));
      tGpuDigis = 
          consumes<CUDAProduct<SiPixelDigisCUDA>>(iConfig.getParameter<edm::InputTag>("heterogeneousPixelDigiClusterSrc"));
      tGpuHits = 
          consumes<CUDAProduct<TrackingRecHit2DCUDA>>(iConfig.getParameter<edm::InputTag>("heterogeneousPixelRecHitSrc"));
  } else {
  }
}

void ClusterTPCUDAdump::analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const {
  if (m_onGPU) {
    auto const& hctp = iEvent.get(tokenGPU_);
    CUDAScopedContextProduce ctx{hctp};

    auto const& ctp = ctx.get(hctp);
    auto const& tpsoa = ctp.view();
    assert(tpsoa.links_d);

    edm::Handle<CUDAProduct<SiPixelDigisCUDA>> gd;
    iEvent.getByToken(tGpuDigis, gd);
    edm::Handle<CUDAProduct<TrackingRecHit2DCUDA>> gh;
    iEvent.getByToken(tGpuHits, gh);

    auto const &gDigis = ctx.get(*gd);
    auto const &gHits = ctx.get(*gh);
    auto ndigis = gDigis.nDigis();
    auto nhits = gHits.nHits();

    if (0 == nhits) return;

    int threadsPerBlock = 256;
    int blocks = (nhits + threadsPerBlock - 1) / threadsPerBlock;
    cudautils::launch(analyzeClusterTP,{threadsPerBlock,blocks,0,ctx.stream()} ,gDigis.view(), ndigis, gHits.view(), nhits, tpsoa);

  } else {
  }
}

void ClusterTPCUDAdump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("onGPU", true);
  desc.add<edm::InputTag>("clusterTP", edm::InputTag("tpClusterProducerCUDAPreSplitting"));
  desc.add<edm::InputTag>("heterogeneousPixelDigiClusterSrc", edm::InputTag("siPixelClustersCUDAPreSplitting"));
  desc.add<edm::InputTag>("heterogeneousPixelRecHitSrc", edm::InputTag("siPixelRecHitsCUDAPreSplitting"));
  descriptions.add("clusterTPCUDAdump", desc);
}

DEFINE_FWK_MODULE(ClusterTPCUDAdump);
