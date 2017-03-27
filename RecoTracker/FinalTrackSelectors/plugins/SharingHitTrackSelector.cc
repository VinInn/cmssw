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
#include "DataFormats/Common/interface/fakearray.h"

volatile void * apointer;

namespace {
  class SharingHitTrackSelector final : public edm::global::EDProducer<> {
   public:
    using FakeProduct = std::vector<fakearray<int,4>>;
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
    auto fakeproduct = std::make_unique<FakeProduct>();
    Product & product = *(Product*)fakeproduct.get();
    
    auto share = 
      [](TrackingRecHit const * it, TrackingRecHit const * jt)->bool { return it->sharesInput(jt,TrackingRecHit::some); };

    Product pairs;
    
    edm::Handle<TkView> trackCollectionHandle;
    evt.getByToken(trackTag,trackCollectionHandle);
    auto const & tracks = *trackCollectionHandle;
    int nt = tracks.size();
    Product::value_type defaultV = {{-1,0,0,0}};
    product.resize(tracks.size(),defaultV);
    for (int i=1; i<nt; ++i) {
      auto const & tk1 = tracks[i];
      auto hb = tk1.recHitsBegin();
      TrackingRecHit const * h1[2] = { (*hb), (*(hb+1)) };
      for (int j=0; j<i; ++j) {
        auto const & tk2 = tracks[j];
        auto hb = tk2.recHitsBegin();
        TrackingRecHit const * h2[2] = { (*hb), (*(hb+1)) };
        for (int k1=0;k1<2;++k1) for (int k2=0;k2<2;++k2) {
          if ( share(h1[k1],h2[k2]) ) {
            Product::value_type v = {{i,j,k1,k2}};
            pairs.push_back(v);
	    product[i][0]=i;
	    product[j][0]=j;
 	    product[i][k1+1]=1;
	    product[j][k2+1]=1;
            break;
          }
        }
      } 
    }

    product.erase(std::remove_if(product.begin(),product.end(),[](Product::value_type const & v){return -1==v[0];}),product.end());
    std::sort(product.begin(),product.end(),[](const auto & a,const auto & b){return a[0]<b[0];});
    std::cout << "size " << product.size() << '/' << tracks.size() << std::endl;
    for (auto const & v : pairs)
      std::cout << v[0] << ','<<v[1]<<": " << v[2] << ','<<v[3]<<": " 
               << tracks[v[0]].numberOfValidHits() << ',' << tracks[v[1]].numberOfValidHits() <<": "
               << tracks[v[0]].charge() << ',' << tracks[v[1]].charge() <<": "
               << tracks[v[0]].pt() << ',' << tracks[v[1]].pt() <<": " << std::endl;
    //                 << (*(tracks[v[0]].recHitsBegin()+v[2]))->globalPosition().perp() << std::endl;
    for (auto const & v : product)
      std::cout << v[0] << ": " <<v[1]<<"," << v[2] << ','<<v[3]<<std::endl;


    // have to make sure "product" is used
    apointer = &product[0];
    
    evt.put(std::move(fakeproduct));

  }


}
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SharingHitTrackSelector);
