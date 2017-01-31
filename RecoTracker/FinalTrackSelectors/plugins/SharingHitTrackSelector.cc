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
    using FakeProduct = std::vector<int>;
    using Product = std::vector<std::array<int,4>>;
    using TkView=edm::View<reco::Track>;
   public:
    explicit SharingHitTrackSelector(const edm::ParameterSet& conf) :
      trackTag(consumes<TkView>(conf.getParameter<edm::InputTag>("src"))) {
      produces<FakeProduct>(); 
    }

   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src",edm::InputTag("generalTracks"));
      descriptions.add("SharingHitTrackSelector", desc);
   }

   virtual void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const override;

   private:
    edm::EDGetTokenT<TkView> trackTag;
  };


  void
  SharingHitTrackSelector::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const {
    auto product = std::make_unique<Product>();

    auto share = 
      [](TrackingRecHit const * it, TrackingRecHit const * jt)->bool { return it->sharesInput(jt,TrackingRecHit::some); };


    edm::Handle<TkView> trackCollectionHandle;
    evt.getByToken(trackTag,trackCollectionHandle);
    auto const & tracks = *trackCollectionHandle;
    int nt = tracks.size();
    for (int i=0; i<nt-1; ++i) {
      auto const & tk1 = tracks[i];
      auto hb = tk1.recHitsBegin();
      TrackingRecHit const * h1[2] = { (*hb), (*(hb+1)) };
      for (int j=i+1; j<nt; ++j) {
        auto const & tk2 = tracks[j];
        auto hb = tk2.recHitsBegin();
        TrackingRecHit const * h2[2] = { (*hb), (*(hb+1)) };
        for (int k1=0;k1<2;++k1) for (int k2=0;k2<2;++k2) {
          if ( share(h1[k1],h2[k2]) ) {
            Product::value_type v = {{i,j,k1,k2}};
            (*product).push_back(v);
            break;
          }
        }
      } 
    }

    std::cout << "size " << product->size() << std::endl;
    for (auto const & v : *product)
      std::cout << v[0] << ','<<v[1]<<": " << v[2] << ','<<v[3]<<": " 
                << (*(tracks[v[0]].recHitsBegin()+v[2]))->globalPosition().perp() << std::endl;


    evt.put(std::move(std::make_unique<FakeProduct>()));
  }


}
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SharingHitTrackSelector);
