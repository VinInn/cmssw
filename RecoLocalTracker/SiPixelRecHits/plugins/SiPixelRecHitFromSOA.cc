#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"


#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"


#include <cuda_runtime.h>

class SiPixelRecHitFromSOA: public edm::stream::EDProducer<edm::ExternalWork> {


public:

  explicit SiPixelRecHitFromSOA(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitFromSOA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<CUDAProduct<TrackingRecHit2DCUDA>> tokenHit_;  // CUDA hits
  edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;  // Legacy Clusters

  uint32_t m_nHits;
  cudautils::host::unique_ptr<uint16_t[]>  m_store16;
  cudautils::host::unique_ptr<float[]>  m_store32;
  cudautils::host::unique_ptr<uint32_t[]> m_hitsModuleStart;
};

SiPixelRecHitFromSOA::SiPixelRecHitFromSOA(const edm::ParameterSet& iConfig):
  tokenHit_(consumes<CUDAProduct<TrackingRecHit2DCUDA>>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
  clusterToken_(consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("src")))
{
    produces<SiPixelRecHitCollectionNew>();
}


void SiPixelRecHitFromSOA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsCUDAPreSplitting"));
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));
  descriptions.add("siPixelRecHitFromSOA",desc);
}


void SiPixelRecHitFromSOA::acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {

  CUDAProduct<TrackingRecHit2DCUDA> const& inputDataWrapped = iEvent.get(tokenHit_);
  CUDAScopedContext ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_nHits = inputData.nHits();

  // std::cout<< "converting " << m_nHits << " Hits"<< std::endl;

  if (0==m_nHits) return;
  m_store32 = inputData.localCoordToHostAsync(ctx.stream());
  //  m_store16 = inputData.detIndexToHostAsync(ctx.stream();
  m_hitsModuleStart = inputData.hitsModuleStartToHostAsync(ctx.stream());
}

void SiPixelRecHitFromSOA::produce(edm::Event& iEvent, edm::EventSetup const& es) {

  auto output = std::make_unique<SiPixelRecHitCollectionNew>();
  if (0==m_nHits) {
    iEvent.put(std::move(output));
    return;
  }

  auto xl = m_store32.get();
  auto yl = xl+m_nHits;
  auto xe = yl+m_nHits;
  auto ye = xe+m_nHits;

  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get( geom );
  geom = geom.product();

  edm::Handle<SiPixelClusterCollectionNew> hclusters;
  iEvent.getByToken(clusterToken_, hclusters);

  auto const & input = *hclusters;

  int numberOfDetUnits = 0;
  int numberOfClusters = 0;
  for (auto DSViter=input.begin(); DSViter != input.end() ; DSViter++) {
    numberOfDetUnits++;
    unsigned int detid = DSViter->detId();
    DetId detIdObject( detid );
    const GeomDetUnit * genericDet = geom->idToDetUnit( detIdObject );
    auto gind = genericDet->index();
    const PixelGeomDetUnit * pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);
    SiPixelRecHitCollectionNew::FastFiller recHitsOnDetUnit(*output, detid);
    auto fc = m_hitsModuleStart[gind];
    auto lc = m_hitsModuleStart[gind+1];
    auto nhits = lc-fc;

    //std::cout << "in det " << gind << "conv " << nhits << " hits from " << DSViter->size() << " legacy clusters" 
    //          <<' '<< lc <<','<<fc<<std::endl;

    if (0==nhits) continue;
    uint32_t ic=0;
    auto jnd = [&](int k) { return fc+k; };
    assert(nhits<=DSViter->size());
    if (nhits!=DSViter->size()) {
       edm::LogWarning("GPUHits2CPU") <<"nhits!= ndigi " << nhits << ' ' << DSViter->size() << std::endl;
    }
    for (auto const & clust : *DSViter) {
      if (ic>=nhits) {
        // FIXME add a way to handle this case, or at least notify via edm::LogError
        break;
      }
      auto ij = jnd(clust.originalId());
      if (ij>=TrackingRecHit2DSOAView::maxHits()) break; // overflow...
      assert(clust.originalId()>=0); assert(clust.originalId()<nhits);
      LocalPoint lp(xl[ij], yl[ij]);
      LocalError le(xe[ij], 0, ye[ij]);
      SiPixelRecHitQuality::QualWordType rqw=0;

      ++ic;
      numberOfClusters++;

      /*   cpu version....  (for reference)
           std::tuple<LocalPoint, LocalError, SiPixelRecHitQuality::QualWordType> tuple = cpe_->getParameters( clust, *genericDet );
           LocalPoint lp( std::get<0>(tuple) );
           LocalError le( std::get<1>(tuple) );
           SiPixelRecHitQuality::QualWordType rqw( std::get<2>(tuple) );
      */

      // Create a persistent edm::Ref to the cluster
      edm::Ref< edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster > cluster = edmNew::makeRefTo(hclusters, &clust);
      // Make a RecHit and add it to the DetSet
      SiPixelRecHit hit( lp, le, rqw, *genericDet, cluster);
      //
      // Now save it =================
      recHitsOnDetUnit.push_back(hit);
      // =============================

      // std::cout << "SiPixelRecHitGPUVI " << numberOfClusters << ' '<< lp << " " << le << std::endl;

    } //  <-- End loop on Clusters


      //  LogDebug("SiPixelRecHitGPU")
      //std::cout << "SiPixelRecHitGPUVI "
      //	<< " Found " << recHitsOnDetUnit.size() << " RecHits on " << detid //;
      // << std::endl;


  } //    <-- End loop on DetUnits

  /*
  std::cout << "SiPixelRecHitGPUVI $ det, clus, lost "
    <<  numberOfDetUnits << ' '
    << numberOfClusters  << ' '
    << std::endl;
  */

  iEvent.put(std::move(output));
}

DEFINE_FWK_MODULE(SiPixelRecHitFromSOA);
