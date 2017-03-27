#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include<cassert>
#include "DataFormats/Common/interface/fakearray.h"

// #include "CommonTools/Utils/interface/DynArray.h"

#include<iostream>
// #define COUT(x) edm::LogVerbatim(x)
#define COUT(x) std::cout<<x<<' '


namespace {

  class TrackCollectionClonerFromMultipleInputs final : public edm::stream::EDProducer<> {
  public:
    using FakeHitList = std::vector<fakearray<int,4>>;
    using HitList = std::vector<std::array<int,4>>;
    using TkView=edm::View<reco::Track>;
    
    explicit TrackCollectionClonerFromMultipleInputs(const edm::ParameterSet& conf):
      trackGenTag(consumes<TkView>(conf.getParameter<edm::InputTag>("genSrc"))),
      trackOriTag(consumes<TkView>(conf.getParameter<edm::InputTag>("oriSrc"))),
      trackRefitTag(consumes<TkView>(conf.getParameter<edm::InputTag>("refitSrc"))),
      m_hitList(consumes<FakeHitList>(conf.getParameter<edm::InputTag>( "hitList" )))
      {
	produces<reco::TrackCollection>();

      }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    
    /// Implementation of produce method
    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    
  private:
    edm::EDGetTokenT<TkView> trackGenTag;  // the collection to clone
    edm::EDGetTokenT<TkView> trackOriTag;  // a subsample of the above
    edm::EDGetTokenT<TkView> trackRefitTag; // the new tracks
    edm::EDGetToken m_hitList;  // the association 3 into 2

  };

 void
 TrackCollectionClonerFromMultipleInputs::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
   edm::ParameterSetDescription desc;
   desc.add<edm::InputTag>("genSrc",edm::InputTag());
   desc.add<edm::InputTag>("oriSrc",edm::InputTag());
   desc.add<edm::InputTag>("refitSrc",edm::InputTag());
   desc.add<edm::InputTag>("hitList",edm::InputTag());
   descriptions.add("TrackCollectionClonerFromMultipleInputs", desc);
 }


  void
  TrackCollectionClonerFromMultipleInputs::produce(edm::Event&evt, const edm::EventSetup&) {
    auto productP = std::make_unique<reco::TrackCollection>();
    auto & product = *productP;
    
    edm::Handle<TkView> track0CollectionHandle;
    evt.getByToken(trackGenTag,track0CollectionHandle);
    auto const & tracksGen = *track0CollectionHandle;
    edm::Handle<TkView> track1CollectionHandle;
    evt.getByToken(trackOriTag,track1CollectionHandle);
    auto const & tracksOri = *track1CollectionHandle;
    edm::Handle<TkView> track2CollectionHandle;
    evt.getByToken(trackRefitTag,track2CollectionHandle);
    auto const & tracksRefit = *track2CollectionHandle;

    edm::Handle<FakeHitList> hitListHandle;
    evt.getByToken(m_hitList, hitListHandle);
    auto const & hitList = *(HitList const *)hitListHandle.product();
    
    COUT("TrackCollectionClonerFromMultipleInputs") << "sizes: "
						    << tracksGen.size() << ' ' << tracksOri.size() << ' ' << tracksRefit.size() << ' '  << hitList.size()<< std::endl;
    
    product.reserve(tracksGen.size());
    if(hitList.size()!=tracksRefit.size()) {
      COUT("TrackCollectionClonerFromMultipleInputs") << "some fit failed..." << std::endl;
      for (auto const&tk:tracksGen) product.push_back(tk);
    } else if (!tracksGen.empty()) {
      auto pG = &tracksGen.front();
      auto eG = &tracksGen.back();
      int nt = tracksRefit.size();
      for (int i=0; i<nt; ++i) {
	auto const & tkO = tracksOri[hitList[i][0]];
	auto const & tkN = tracksRefit[i];
	auto pN= &tkO;
	assert(pN>=pG && pN<=eG);
	for(;pG<pN; ++pG) product.push_back(*pG);
	product.push_back(tkN);
	assert(pN==pG);
	++pG;
      }
      assert(pG<=(eG+1));
      for(;pG<=eG; ++pG) product.push_back(*pG);
    }

    assert(product.size()==tracksGen.size());

    evt.put(std::move(productP));
    
    
  }

}

DEFINE_FWK_MODULE(TrackCollectionClonerFromMultipleInputs);


