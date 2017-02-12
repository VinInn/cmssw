// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

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


#include<iostream>
// #define COUT(x) edm::LogVerbatim(x)
#define COUT(x) std::cout<<x<<' '


namespace {
  class TwoTrackAnalyzer final : public edm::stream::EDAnalyzer<> {
  public:
    using FakeHitList = std::vector<fakearray<int,4>>;
    using HitList = std::vector<std::array<int,4>>;
    using TkView=edm::View<reco::Track>;
    explicit TwoTrackAnalyzer(const edm::ParameterSet& conf) :
      track1Tag(consumes<TkView>(conf.getParameter<edm::InputTag>("oriSrc"))),
      track2Tag(consumes<TkView>(conf.getParameter<edm::InputTag>("refitSrc"))),
      m_hitList(consumes<FakeHitList>(conf.getParameter<edm::InputTag>( "hitList" )))

      {}
    
    ~TwoTrackAnalyzer(){}
    
    
  private:
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    
    
    edm::EDGetTokenT<TkView> track1Tag;
    edm::EDGetTokenT<TkView> track2Tag;
    edm::EDGetToken m_hitList;

  };
  
  void
  TwoTrackAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup&)
  {
    edm::Handle<TkView> track1CollectionHandle;
    evt.getByToken(track1Tag,track1CollectionHandle);
    auto const & tracks1 = *track1CollectionHandle;
    edm::Handle<TkView> track2CollectionHandle;
    evt.getByToken(track2Tag,track2CollectionHandle);
    auto const & tracks2 = *track2CollectionHandle;

    edm::Handle<FakeHitList> hitListHandle;
    evt.getByToken(m_hitList, hitListHandle);
    auto const & hitList = *(HitList const *)hitListHandle.product();

   
    COUT("2TrackA") << "sizes: " << tracks1.size() << ' '  << tracks2.size() << ' '  << hitList.size()<< std::endl;

    if(hitList.size()!=tracks2.size()) { COUT("2TrackA") << "some fit failed..." << std::endl; return;}
    int nt = tracks2.size();
    for (int i=0; i<nt; ++i) {
      auto const & tk1 = tracks1[hitList[i][0]];
      auto const & tk2 = tracks2[i];
      COUT("2TrackA") << "nh/pt/dxy " << tk1.numberOfValidHits()  << ',' << tk2.numberOfValidHits() << ' '
		      << tk1.pt() << ',' << tk2.pt() << ' ' << tk1.dxy() << ',' << tk2.dxy() << std::endl;
    }
    
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(TwoTrackAnalyzer);
