#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseParams.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"


#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseDim.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseSigma.h"


#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include<mutex>


namespace {

  struct Stat{
    static constexpr double fact = 1000.;


    Stat() : 
      nhp(0), ntt(0), ntp9(0), ntp12(0), ntp16(0),
      n(0),zx(0),zy(0),zx2(0),zy2(0) {}
    ~Stat() {
      auto x = double(zx)/fact/n;
      auto y = double(zy)/fact/n;
     
      auto sx = std::sqrt(double(zx2)/fact/n -x*x);
      auto sy = std::sqrt(double(zy2)/fact/n -y*y);
 
     
      std::cout << "ClusEllipse Stat " << ntt << ' ' << ntp9/double(ntt) << '/' << ntp12/double(ntt) << '/' << ntp16/double(ntt) 
                << ' ' << n << ' ' << nhp/double(n)
                << ' ' << x << '/' << y
       	       	<< ' ' << sx << '/' << sy
                << std::endl;

     }

    std::atomic<long long> nhp, ntt, ntp9, ntp12, ntp16;
    std::atomic<long long> n, zx,zy,zx2,zy2;
  }; 

  Stat stat;

  std::mutex theMutex;

}


namespace {

  class ClusEllipseAnalyzer final : public edm::global::EDAnalyzer<> {

  public : 

   explicit ClusEllipseAnalyzer(const edm::ParameterSet& pset);
   void analyze(edm::StreamID, const edm::Event& evt, const edm::EventSetup& es) const override;
   void endJob() override;


  private:

   const edm::EDGetTokenT<reco::TrackCollection> tracks_token;


  };


  ClusEllipseAnalyzer::ClusEllipseAnalyzer(const edm::ParameterSet& pset) :
  tracks_token(consumes<reco::TrackCollection>(pset.getParameter<edm::InputTag>("tracks"))) {

  }

  void ClusEllipseAnalyzer::analyze(edm::StreamID, const edm::Event& evt, const edm::EventSetup& es) const {

    ClusEllipseDim dnnD;
    ClusEllipseSigma dnnS;


    edm::ESHandle<TrackerTopology> tTopoHandle;
    es.get<TrackerTopologyRcd>().get(tTopoHandle);
    auto const & tkTpl = *tTopoHandle;


    // Get tracks
    edm::Handle<reco::TrackCollection> tracks;
    evt.getByToken(tracks_token, tracks);
    
    for (auto const & track : *tracks) {
      if (!track.quality(reco::Track::highPurity)) continue;
      if (track.numberOfValidHits()<4) continue;
      if (track.pt()<0.2) continue;
      auto const & trajParams = track.extra()->trajParams();
      assert(trajParams.size()==track.recHitsSize());
      auto hb = track.recHitsBegin();
      int nh=0; float chi2=0;
      for(unsigned int h=0;h<track.recHitsSize();h++){
        auto recHit = (BaseTrackerRecHit const *)(*(hb+h));   
        if (!recHit->isValid()) continue;
        if (!recHit->isPixel()) continue;

        auto id = recHit->geographicalId();
 
        bool isBarrel = id.subdetId() == PixelSubdetector::PixelBarrel;

        float thickness = isBarrel ? 0.0285f : 0.029f;
        constexpr float ipx = 1.f/0.01f; constexpr float ipy = 1.f/0.015f;  // phase1 pitch

        auto const & ltp = trajParams[h];
        auto tkdx = ipx*thickness * ltp.momentum().x()/ltp.momentum().z();
        auto tkdy = ipy*thickness * ltp.momentum().y()/ltp.momentum().z();
        if( tkdy<0) { tkdx = -tkdx;}
        tkdy = std::abs(tkdy);

        auto const & pRecHit = *(reinterpret_cast<SiPixelRecHit const *>(recHit));

        ClusEllipseParams cep; cep.fill(pRecHit,tkTpl);
        if (cep.m_layer==0) continue;
       	memcpy(dnnD.arg0_data(),cep.data(),9*4);
        memcpy(dnnS.arg0_data(),dnnD.arg0_data(),9*4);
        dnnD.Run();
        dnnS.Run();

        tkdx = cep.m_sy>1 ? tkdx : std::abs(tkdx);

        auto pdx = 0.25f *(cep.m_dx + dnnD.result0_data()[0]);
       	auto pdy = cep.m_dy + dnnD.result0_data()[1];

        auto psx = dnnS.result0_data()[0];
       	auto psy = dnnS.result0_data()[1];

        auto zx = (pdx-tkdx)/psx;
       	auto zy	= (pdy-tkdy)/psy;

        /*
        { 
         using Lock = std::unique_lock<std::mutex>;
         Lock lock(theMutex);
         for (int i=0; i<9; ++i) std::cout << cep.data()[i] << ' ';
         std::cout << pdx << '/' << pdy << ' ' << psx << '/' << psy
                   << ' ' << tkdx << '/' << tkdy << std::endl; 
        }
        */     

        constexpr float fact = Stat::fact;
        ++stat.n;
        stat.zx += fact*zx;
        stat.zy	+= fact*zy;
       	stat.zx2 += fact*zx*zx;
        stat.zy2 += fact*zy*zy;
        if (std::abs(zx)<4 && std::abs(zy)<4) ++stat.nhp;
        chi2+=zx*zx+zy*zy;
        ++nh; 

      } // hit loop
      ++stat.ntt;
      if (nh==0 || chi2 < 18.f*float(nh) ) ++stat.ntp9;
      if (nh==0 || chi2 < 24.f*float(nh) ) ++stat.ntp12;
      if (nh==0 || chi2 < 32.f*float(nh) ) ++stat.ntp16;
    }  // track loop

  }

  void ClusEllipseAnalyzer::endJob(){
  }






} // namespace



#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ClusEllipseAnalyzer);
