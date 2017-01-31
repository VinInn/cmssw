#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"


#include<vector>
#include<memory>
#include<cassert>
namespace {
  class SharingHitTrackSelector final : public edm::global::EDProducer<> {
   public:
    using Product = std::vector<int>;
    using TkView=edm::View<reco::Track>;
   public:
    explicit SharingHitTrackSelector(const edm::ParameterSet& conf) :
      trackTag(consumes<TkView>(conf.getParameter<edm::InputTag>("src"))) {
      produces<Product>(); 
    }

   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src",edm::InputTag());
      descriptions.add("SharingHitTrackSelector", desc);
   }

   virtual void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const override;

   private:
    edm::EDGetTokenT<TkView> trackTag;
  };


  void
  SharingHitTrackSelector::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const {
    auto product = std::make_unique<Product>();

   Handle<TkView> trackCollectionHandle;
   iEvent.getByToken(trackTag,trackCollectionHandle);
   auto const & tracks = *trackCollectionHandle;
   auto nt = tracks.size();
   if (nt>1)
   for (unsigned int i=0; i<nt-1; ++i) {
     auto const & tk1 = tracks[i];
     
     for (unsigned int j=i+1; j<nt; ++j) {
 
   }


    evt.put(std::move(product));
  }


}
