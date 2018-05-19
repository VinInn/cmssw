// VI January 2012: needs to be migrated to use cluster directly

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#

#include "TROOT.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

#include <map>
#include <vector>
#include <fstream>

#include<mutex>

using namespace std;

// pixel
#define exMax 10
#define eyMax 15


namespace {

  const std::set<unsigned> RelevantProcesses = { 0 };
    //const std::set<unsigned> RelevantProcesses = { 2, 7, 9, 11, 13, 15 };


  struct TkParam { int n=0; float dx,dy;};

  std::atomic<int> gevid(1);

}


/*****************************************************************************/
class PixelClusterShapeExtractor final : public edm::global::EDAnalyzer<>
{
 public:
   explicit PixelClusterShapeExtractor(const edm::ParameterSet& pset);
   void analyze(edm::StreamID, const edm::Event& evt, const edm::EventSetup&) const override;
   void endJob() override;

 private:

   void init();

   bool isSuitable(const PSimHit & simHit, const GeomDetUnit & gdu) const;

   // Sim
   void processSim(int evid, const SiPixelRecHit &   recHit, ClusterShapeHitFilter const & theClusterFilter,
                   const PSimHit & simHit, const SiPixelClusterShapeCache& clusterShapeCache, 
                   const TkParam * tkParam, const TrackerTopology & tkTpl,
                   const vector<TH2F *> & histo) const;

   // Rec
   void processRec(float pt, long long tid, const SiPixelRecHit &   recHit, ClusterShapeHitFilter const & theFilter,
                   LocalPoint loc, LocalVector ldir, const SiPixelClusterShapeCache& clusterShapeCache,
                   const TkParam * tkParam,
                   const TrackerTopology & tkTpl,
                   const vector<TH2F *> & histo) const;

   bool checkSimHits
    (const TrackingRecHit & recHit, TrackerHitAssociator const & theAssociator,
     PSimHit & simHit, pair<unsigned int, float> & key, unsigned int & ss) const;

   void processPixelRecHits
     (SiPixelRecHitCollection::DataContainer const & recHits, 
      TrackerHitAssociator const & theAssociator,
      ClusterShapeHitFilter const & theFilter, 
      SiPixelClusterShapeCache const & clusterShapeCache,
      const TkParam * tkParam,
      const TrackerTopology & tkTpl) const;

    void processRecTracks
      (const edm::Event& ev, const edm::EventSetup& es, TkParam * tkParam) const;

   void analyzeSimHits  (int sid, const edm::Event& ev, const edm::EventSetup& es) const;
   void analyzeRecTracks(const edm::Event& ev, const edm::EventSetup& es) const;

   using Lock = std::unique_lock<std::mutex>; 
   TFile * file;
   mutable ofstream csvFile;
   mutable ofstream csvHitFile;

   const bool hasSimHits;
   const bool hasRecTracks;
   const bool noBPIX1;

   const edm::EDGetTokenT<reco::TrackCollection> tracks_token;
   const edm::EDGetTokenT<edmNew::DetSetVector<SiPixelRecHit>> pixelRecHits_token;
   const edm::EDGetTokenT<SiPixelClusterShapeCache> clusterShapeCache_token;
   const    TrackerHitAssociator::Config trackerHitAssociatorConfig_;

   std::unique_ptr<std::mutex[]> theMutex;
   std::vector<TH2F *> hspc; // simulated pixel cluster
   std::vector<TH2F *> hrpc; // reconstructed pixel cluster
};

/*****************************************************************************/
void PixelClusterShapeExtractor::init()
{
  // Declare histograms
  char histName[256];

  // pixel
  for(int subdet = 0; subdet <= 1; subdet++)
  {
    for(int ex = 0; ex <= exMax; ex++)
    for(int ey = 0; ey <= eyMax; ey++)
    {
      sprintf(histName,"hspc_%d_%d_%d",subdet, ex,ey);
      hspc.push_back(new TH2F(histName,histName, 
                         10 * 2 * (exMax+2), -(exMax+2),(exMax+2),
                         10 * 2 * (eyMax+2), -(eyMax+2),(eyMax+2)));

      sprintf(histName,"hrpc_%d_%d_%d",subdet, ex,ey);
      hrpc.push_back(new TH2F(histName,histName, 
                         10 * 2 * (exMax+2), -(exMax+2),(exMax+2),
                         10 * 2 * (eyMax+2), -(eyMax+2),(eyMax+2)));
    }
  }
  theMutex.reset(new std::mutex[hspc.size()]);
}

/*****************************************************************************/
PixelClusterShapeExtractor::PixelClusterShapeExtractor(const edm::ParameterSet& pset) :
  csvFile("clusterShape.csv"),
  csvHitFile("Hits.csv"),
  hasSimHits(pset.getParameter<bool>("hasSimHits")),
  hasRecTracks(pset.getParameter<bool>("hasRecTracks")),
  noBPIX1(pset.getParameter<bool>("noBPIX1")),
  tracks_token(consumes<reco::TrackCollection>(pset.getParameter<edm::InputTag>("tracks"))),
  pixelRecHits_token(consumes<edmNew::DetSetVector<SiPixelRecHit>>(edm::InputTag("siPixelRecHits"))),
  clusterShapeCache_token(consumes<SiPixelClusterShapeCache>(pset.getParameter<edm::InputTag>("clusterShapeCacheSrc"))),
  trackerHitAssociatorConfig_(pset, consumesCollector())
{
  csvFile << "isBarrel layer simX simY simSX simSY recX recY tkSX tkSY x y xx yy xy dx dy l2 sx sy s q" << std::endl;
  csvHitFile << "trackID pt r z phi isBarrel layer" << std::endl; 
  file = new TFile("clusterShape.root","RECREATE");
  file->cd();
  init();
}

/*****************************************************************************/
void PixelClusterShapeExtractor::endJob()
{

  file->cd();

  // simulated
  for(auto h = hspc.begin(); h!= hspc.end(); h++) (*h)->Write();

  // reconstructed
  for(auto h = hrpc.begin(); h!= hrpc.end(); h++) (*h)->Write();

  file->Close();
}


/*****************************************************************************/
bool PixelClusterShapeExtractor::isSuitable(const PSimHit & simHit, const GeomDetUnit & gdu) const
{
  // Outgoing?
  // very expensive....
  GlobalVector gvec = gdu.position() -
                      GlobalPoint(0,0,0);
  LocalVector  lvec = gdu.toLocal(gvec);
  LocalVector  ldir = simHit.exitPoint() - simHit.entryPoint();
  
  // cut on size as well (pixel is 285um thick...
  bool isOutgoing = std::abs(ldir.z())>0.02f && (lvec.z()*ldir.z() > 0); 

    ///  ?????
  const bool isRelevant = RelevantProcesses.count(simHit.processType());
  // From a relevant process? primary or decay
  //bool isRelevant = (simHit.processType() == 2 ||
  //                   simHit.processType() == 4);

  constexpr float ptCut2 = 0.2*0.2; //  0.050*0.050;
  // Fast enough? pt > 50 MeV/c   FIXME (at least 200MeV....
  auto sp = gdu.toGlobal(simHit.momentumAtEntry());
  bool isFast = (sp.perp2() > ptCut2);

    //std::cout << "isOutgoing = " << isOutgoing << ", isRelevant = " << simHit.processType() << ", isFast = " << isFast << std::endl;
  return (isOutgoing && isRelevant && isFast);
}

/*****************************************************************************/
void PixelClusterShapeExtractor::processRec(float spt, long long tid, const SiPixelRecHit & recHit, ClusterShapeHitFilter const & theClusterShape,
    LocalPoint loc, LocalVector ldir, const SiPixelClusterShapeCache& clusterShapeCache,    const TkParam * tkParam,
    const TrackerTopology & tkTpl, const vector<TH2F *> & histo) const
{
  int part;
  ClusterData::ArrayType meas;
  pair<float,float> pred;
 
  if(theClusterShape.getSizes(recHit,ldir,clusterShapeCache, part,meas,pred))
   if(meas.size() == 1)
    if(meas.front().first  <= exMax && 
       meas.front().second <= eyMax)
    {
      int i = (part * (exMax + 1) +
               meas.front().first) * (eyMax + 1) +
               meas.front().second;
#ifdef DO_DEBUG
      if (meas.front().second==0 && std::abs(pred.second)>3)
      {
        Lock lock(theMutex[0]);
        int id = recHit.geographicalId();
        std::cout << id << " bigpred " << meas.front().first << '/'<<meas.front().second 
                  << ' ' << pred.first << '/' << pred.second << ' ' << ldir << ' ' << ldir.mag()<< std::endl;
      }
#endif
     std::ostringstream hcsv;

     bool isBarrel = (recHit.geographicalId().subdetId() == int(PixelSubdetector::PixelBarrel));
     auto gpos =  recHit.globalPosition();
     hcsv << tid << ' ' << spt << ' '
          << gpos.perp() << ' ' << gpos.z() << ' ' << gpos.phi() << ' '
          <<  isBarrel << ' ' << tkTpl.layer(recHit.geographicalId());
     {
      Lock llock(theMutex[0]);
      csvHitFile << hcsv.str() << std::endl;
     }

     auto ck = ((BaseTrackerRecHit const &)(recHit)).firstClusterRef().key();
   
     auto tp = tkParam[ck];
     // if (tp.n!=1) return;

      auto const clus = *recHit.cluster();
      auto const& topol = reinterpret_cast<const PixelGeomDetUnit*>(recHit.detUnit())->specificTopology();
      if (clus.minPixelCol()==0) return;
      if (clus.maxPixelCol()+1==topol.ncolumns()) return;
      if (clus.minPixelRow()==0) return;
      if (clus.maxPixelRow()+1==topol.nrows()) return;
      if (clus.minPixelRow()<=79 && clus.maxPixelRow()>=80) return;

      auto dc = 52-clus.minPixelCol()%52;
      if (dc==52) return;
      bool hasB = dc==52 || clus.size()>=dc;

      std::ostringstream csv;
//      std::cout << "pitch " << recHit.det()->specificTopology().pitch().first << std::endl;
      constexpr float ipx = 1.f/0.01f; constexpr float ipy = 1.f/0.015f;
      auto dx = ldir.x()*std::copysign(ipx,ldir.z());
      auto dy = ldir.y()*std::copysign(ipy,ldir.z());
      if (dy<0) {dx = -dx;} dy = std::abs(dy);   
      csv << isBarrel << ' ' << tkTpl.layer(recHit.geographicalId()) << ' ' << loc.x() << ' ' << loc.y() 
          << ' ' << dx << ' ' << dy;
      csv << ' ' << recHit.localPosition().x() << ' ' <<  recHit.localPosition().y();

       if (tp.n==1) {
        dx = tp.dx*ipx;
        dy = tp.dy*ipy;
        if (dy<0) {dx = -dx;} dy = std::abs(dy);
      } else { dx = dy = -999.f;}
      csv << ' ' << dx << ' ' << dy;      

      float qx=0, qy=0, q2x=0, q2y=0 ,qxy=0, q=0;     
      int isize = clus.pixelADC().size();
      
      for (int i=0; i<isize; ++i) {
        float yo = clus.pixelOffset()[i*2+1]<dc ? 0.0f : ( clus.pixelOffset()[i*2+1]==dc ? 0.5f : 1.f);
        auto c = float(clus.pixelADC()[i]);
        auto x = float(clus.pixelOffset()[i*2]);
       	auto y = float(clus.pixelOffset()[i*2+1])+yo;
        q+=c; qx+=c*x;qy+=c*y; 
        q2x+=c*x*x; q2y+=c*y*y;
        qxy+=c*x*y;
      }
      qx /=q; qy /=q; 
      q2x = q2x/q - qx*qx; q2y = q2y/q - qy*qy;
      qxy = qxy/q - qx*qy;
      auto tr = q2x+q2y; auto det = q2x*q2y-qxy*qxy;
      auto l1 = 0.5f*(tr + std::sqrt(tr*tr-4.f*det));
      auto l2 = l1>0 ? det/l1 : 0;
      auto ly = q2y>0 ? l1-q2x : 0.f; auto lx = q2y>0 ?  qxy : 1.f; auto norm = 4.f*std::sqrt(l1/(lx*lx+ly*ly));
      // auto ll = 4.f*std::sqrt(l1);
      lx *=norm; ly*=norm;
      csv << ' ' << qx << ' ' << qy << ' ' << q2x << ' ' << q2y << ' ' << qxy;
      csv << ' ' << lx << ' ' << ly << ' ' << 4.f*sqrt(l2);
      csv << ' ' << clus.sizeX() << ' ' << clus.sizeY() + (hasB?1:0) << ' ' << clus.size() << ' ' << q;
      {
        Lock lock(theMutex[i]);
        histo[i]->Fill(pred.first, pred.second);
      }
      Lock lock(theMutex[0]);
      csvFile << csv.str() << std::endl;
    }
}

/*****************************************************************************/
void PixelClusterShapeExtractor::processSim(int evid, const SiPixelRecHit & recHit, ClusterShapeHitFilter const & theClusterFilter,
     const PSimHit & simHit, const SiPixelClusterShapeCache& clusterShapeCache, const TkParam * tkParam,
     const TrackerTopology & tkTpl, const vector<TH2F *> & histo) const
{
  auto const & gdu = *recHit.detUnit();
  LocalVector ldir = simHit.exitPoint() - simHit.entryPoint(); 
  LocalPoint loc(0.5f*(simHit.exitPoint().basicVector() + simHit.entryPoint().basicVector()));
  assert(simHit.trackId()<100000);
  long long tid = 100000*evid+simHit.trackId();
  auto sp = gdu.toGlobal(simHit.momentumAtEntry());

  float spt = sp.perp();
  processRec(spt, tid, recHit, theClusterFilter, loc, ldir, clusterShapeCache, tkParam, tkTpl, histo);
}

/*****************************************************************************/
bool PixelClusterShapeExtractor::checkSimHits
  (const TrackingRecHit & recHit, TrackerHitAssociator const & theHitAssociator,
   PSimHit & simHit, pair<unsigned int, float> & key, unsigned int & ss) const
{
  auto const & simHits = theHitAssociator.associateHit(recHit);

    //std::cout << "simHits.size() = " << simHits.size() << std::endl;
  for (auto const & sh : simHits)
  {
    if(isSuitable(sh, *recHit.detUnit()))
    {
      simHit = sh; 
      key = {simHit.trackId(),simHit.timeOfFlight()};
      ss = simHits.size();
      return true;
    }
  } 

  return false;
}

/*****************************************************************************/
void PixelClusterShapeExtractor::processPixelRecHits(
   const SiPixelRecHitCollection::DataContainer & recHits,
   TrackerHitAssociator const & theHitAssociator, 
   ClusterShapeHitFilter const & theFilter,
   const SiPixelClusterShapeCache& clusterShapeCache,
   const TkParam * tkParam,
   const TrackerTopology & tkTpl) const
{

  int evid = 0;
  {
    Lock lock(theMutex[0]);
    evid = gevid;
    gevid++;
  }


  struct Elem { const SiPixelRecHit * rhit; PSimHit shit; unsigned int size;};
  std::map<pair<unsigned int, float>, Elem> simHitMap;

  PSimHit simHit;
  pair<unsigned int, float> key;
  unsigned int ss;

  for(auto const & recHit : recHits) {
    if(noBPIX1 && tkTpl.pxbLayer(recHit.geographicalId())==1) continue;
    if(!checkSimHits(recHit, theHitAssociator, simHit, key,ss)) continue;
          // Fill map
          if(simHitMap.count(key) == 0)
              { simHitMap[key] = {&recHit,simHit,ss}; }
          else if( recHit.cluster()->size() >
                   simHitMap[key].rhit->cluster()->size())
                   simHitMap[key] = {&recHit,simHit,std::max(ss,simHitMap[key].size)};
  }
  for (auto const & elem : simHitMap)  {
   /* irrelevant
   auto const rh = *elem.second.rhit;
   auto const& topol = reinterpret_cast<const PixelGeomDetUnit*>(rh.detUnit())->specificTopology();
   auto const & cl = *rh.cluster();
   if (cl.minPixelCol()==0) continue;
   if (cl.maxPixelCol()+1==topol.ncolumns()) continue;
   if (cl.minPixelRow()==0) continue; 
   if (cl.maxPixelRow()+1==topol.nrows()) continue;
   */
   if (elem.second.size==1)
       processSim(evid, *elem.second.rhit, theFilter, elem.second.shit, clusterShapeCache, tkParam, tkTpl, hspc);
  }
}




/*****************************************************************************/
void PixelClusterShapeExtractor::analyzeSimHits
  (int sid, const edm::Event& ev, const edm::EventSetup& es) const
{
  edm::ESHandle<ClusterShapeHitFilter> shape;
  es.get<CkfComponentsRecord>().get("ClusterShapeHitFilter",shape);
  auto const & theClusterShape = *shape.product();

  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  auto const & tkTpl = *tTopoHandle;


  edm::Handle<SiPixelClusterShapeCache> clusterShapeCache;
  ev.getByToken(clusterShapeCache_token, clusterShapeCache);

  // Get associator
  auto theHitAssociator = std::make_unique<TrackerHitAssociator>(ev,trackerHitAssociatorConfig_);

  // Pixel hits
  {
    edm::Handle<SiPixelRecHitCollection> coll;
    ev.getByToken(pixelRecHits_token, coll);

    edm::Handle<SiPixelClusterShapeCache> clusterShapeCache;
    ev.getByToken(clusterShapeCache_token, clusterShapeCache);

    auto const & recHits = coll.product()->data();
    TkParam tkParam[recHits.size()];
    processRecTracks(ev,es,tkParam);
    processPixelRecHits(recHits, *theHitAssociator, theClusterShape, *clusterShapeCache, tkParam, tkTpl);
  }

}

/*****************************************************************************/
void PixelClusterShapeExtractor::processRecTracks
  (const edm::Event& ev, const edm::EventSetup& es, TkParam * tkParam) const
{
  
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  auto const & tkTpl = *tTopoHandle;


  // Get tracks
  edm::Handle<reco::TrackCollection> tracks;
  ev.getByToken(tracks_token, tracks);

 for (auto const & track : *tracks)
 {
    if (!track.quality(reco::Track::highPurity)) continue;
    if (track.numberOfValidHits()<4) continue;
    if (track.pt()<0.2) continue;
    auto const & trajParams = track.extra()->trajParams();
    assert(trajParams.size()==track.recHitsSize());
    auto hb = track.recHitsBegin();
    for(unsigned int h=0;h<track.recHitsSize();h++){
      auto recHit = (BaseTrackerRecHit const *)(*(hb+h));   
      if (!recHit->isValid()) continue;
      if (!recHit->isPixel()) continue;

      auto id = recHit->geographicalId();
      if(noBPIX1 && tkTpl.pxbLayer(id)==1) continue;
 
 
      bool isBarrel = id.subdetId() == PixelSubdetector::PixelBarrel;

      float thickness = isBarrel ? 0.0285f : 0.029f;

      auto const & ltp = trajParams[h];

      auto ck = recHit->firstClusterRef().key();
      auto & tp = tkParam[ck];
      ++tp.n;
      tp.dx = thickness * ltp.momentum().x()/ltp.momentum().z();
      tp.dy = thickness * ltp.momentum().y()/ltp.momentum().z();

    }
  }
}



/*****************************************************************************/
void PixelClusterShapeExtractor::analyzeRecTracks
  (const edm::Event& ev, const edm::EventSetup& es) const
{
/*
  edm::ESHandle<ClusterShapeHitFilter> shape;
  es.get<CkfComponentsRecord>().get("ClusterShapeHitFilter",shape);
  auto const & theClusterShape = *shape.product();

  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  auto const & tkTpl = *tTopoHandle;  


  // Get tracks
  edm::Handle<reco::TrackCollection> tracks;
  ev.getByToken(tracks_token, tracks);

  edm::Handle<SiPixelClusterShapeCache> clusterShapeCache;
  ev.getByToken(clusterShapeCache_token, clusterShapeCache);

 
 for (auto const & track : *tracks) 
 {
    if (!track.quality(reco::Track::highPurity)) continue;
    if (track.numberOfValidHits()<8) continue;
    auto const & trajParams = track.extra()->trajParams();
    assert(trajParams.size()==track.recHitsSize());
    auto hb = track.recHitsBegin();
    for(unsigned int h=0;h<track.recHitsSize();h++){
      auto recHit = *(hb+h);
      if (!recHit->isValid()) continue;
      auto id = recHit->geographicalId();
      if(noBPIX1 && tkTpl.pxbLayer(id)==1) continue;

      // check that we are in the pixel
      auto subdetid = (id.subdetId());
      bool isPixel = subdetid == PixelSubdetector::PixelBarrel || subdetid == PixelSubdetector::PixelEndcap;
      
      auto const & ltp = trajParams[h];
      
      auto ldir = ltp.momentum()/ltp.momentum().mag();

      if(isPixel) 
      {
        // Pixel
        const SiPixelRecHit* pixelRecHit =
          dynamic_cast<const SiPixelRecHit *>(recHit);

        if(pixelRecHit != nullptr)
          processRec(0, 0, *pixelRecHit, theClusterShape, LocalPoint(), ldir, *clusterShapeCache, tkTpl, hrpc);
      }
    }
  }
*/

}

/*****************************************************************************/
void PixelClusterShapeExtractor::analyze
  (edm::StreamID sid,const edm::Event& ev, const edm::EventSetup& es) const
{
  if(hasSimHits)
  {
    LogTrace("MinBiasTracking")
      << " [ClusterShape] analyze simHits, recHits";
    analyzeSimHits(sid.value(), ev, es);
  } 

  if(hasRecTracks)
  {
    LogTrace("MinBiasTracking") 
      << " [ClusterShape] analyze recHits on recTracks";
    analyzeRecTracks(ev,es);
  } 
}

DEFINE_FWK_MODULE(PixelClusterShapeExtractor);

