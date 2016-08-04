#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"


#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"


namespace {

  class ClusterChargeMasker : public edm::stream::EDProducer<> {
  public:
    ClusterChargeMasker(const edm::ParameterSet& iConfig);
    ~ClusterChargeMasker(){}
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override ;
  private:
 

    using PixelMaskContainer = edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster>>;
    using StripMaskContainer = edm::ContainerMask<edmNew::DetSetVector<SiStripCluster>>;

    const float minBadStripCharge_;
    const float minGoodStripCharge_;


    const edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
    const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClusters_;

    edm::EDGetTokenT<PixelMaskContainer> oldPxlMaskToken_;
    edm::EDGetTokenT<StripMaskContainer> oldStrMaskToken_;



  };

  void ClusterChargeMasker::
  fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    edm::ParameterSetDescription descCCC = getFilledConfigurationDescription4CCC();
    desc.add<edm::ParameterSetDescription>("clusterChargeCut", descCCC);
    desc.add<edm::ParameterSetDescription>("clusterChargeCutForGoodAPV", descCCC);
    desc.add<edm::InputTag>("pixelClusters",edm::InputTag("siPixelClusters"));
    desc.add<edm::InputTag>("stripClusters",edm::InputTag("siStripClusters"));
    desc.add<edm::InputTag>("oldClusterRemovalInfo",edm::InputTag());
    descriptions.add("ClusterChargeMasker", desc);
  }


  ClusterChargeMasker::ClusterChargeMasker(const edm::ParameterSet& iConfig) :
    minBadStripCharge_(clusterChargeCut(iConfig)),
    minGoodStripCharge_(std::max(clusterChargeCut(iConfig,"clusterChargeCutForGoodAPV"),minBadStripCharge_)),
    pixelClusters_(consumes<edmNew::DetSetVector<SiPixelCluster>>(iConfig.getParameter<edm::InputTag>("pixelClusters"))),
    stripClusters_(consumes<edmNew::DetSetVector<SiStripCluster>>(iConfig.getParameter<edm::InputTag>("stripClusters")))
  {
    produces<edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster>>>();
    produces<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster>>>();

    auto const &  oldClusterRemovalInfo = iConfig.getParameter<edm::InputTag>("oldClusterRemovalInfo");
    if (!oldClusterRemovalInfo.label().empty()) {
      oldPxlMaskToken_ = consumes<PixelMaskContainer>(oldClusterRemovalInfo);
      oldStrMaskToken_ = consumes<StripMaskContainer>(oldClusterRemovalInfo);
    }

  }


  void
  ClusterChargeMasker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {

 
    edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
    iEvent.getByToken(pixelClusters_, pixelClusters);
    edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
    iEvent.getByToken(stripClusters_, stripClusters);


    std::vector<bool> collectedStrips;
    std::vector<bool> collectedPixels;

    if(!oldPxlMaskToken_.isUninitialized()) {
      edm::Handle<PixelMaskContainer> oldPxlMask;
      edm::Handle<StripMaskContainer> oldStrMask;
      iEvent.getByToken(oldPxlMaskToken_ ,oldPxlMask);
      iEvent.getByToken(oldStrMaskToken_ ,oldStrMask);
      LogDebug("ClusterChargeMasker")<<"to merge in, "<<oldStrMask->size()<<" strp and "<<oldPxlMask->size()<<" pxl";
      oldStrMask->copyMaskTo(collectedStrips);
      oldPxlMask->copyMaskTo(collectedPixels);
      assert(stripClusters->dataSize()>=collectedStrips.size());
      collectedStrips.resize(stripClusters->dataSize(), false);
    }else {
      collectedStrips.resize(stripClusters->dataSize(), false);
      collectedPixels.resize(pixelClusters->dataSize(), false);
    } 

    auto const & clusters = stripClusters->data();
    for (auto const & item : stripClusters->ids()) {
	
	if (!item.isValid()) continue;  // not umpacked  (hlt only)
	
	auto detid = item.id;
	
	for (int i = item.offset; i<item.offset+int(item.size); ++i) {
          auto clusCharge = siStripClusterTools::chargePerCM(detid,clusters[i]);
	  if(clusCharge < minGoodStripCharge_) collectedStrips[i] = true; 
	}

    }

    std::auto_ptr<StripMaskContainer> removedStripClusterMask(
         new StripMaskContainer(edm::RefProd<edmNew::DetSetVector<SiStripCluster> >(stripClusters),collectedStrips));
      LogDebug("ClusterChargeMasker")<<"total strip to skip: "<<std::count(collectedStrips.begin(),collectedStrips.end(),true);
      // std::cout << "ClusterChargeMasker " <<"total strip to skip: "<<std::count(collectedStrips.begin(),collectedStrips.end(),true) 
      //          << " for CCC " << minGoodStripCharge_ <<std::endl;
       iEvent.put( removedStripClusterMask );

      std::auto_ptr<PixelMaskContainer> removedPixelClusterMask(
         new PixelMaskContainer(edm::RefProd<edmNew::DetSetVector<SiPixelCluster> >(pixelClusters),collectedPixels));      
      LogDebug("ClusterChargeMasker")<<"total pxl to skip: "<<std::count(collectedPixels.begin(),collectedPixels.end(),true);
      iEvent.put( removedPixelClusterMask );
 


  }


}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ClusterChargeMasker);

