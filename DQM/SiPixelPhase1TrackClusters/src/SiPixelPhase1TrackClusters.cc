// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackClusters
// Class:       SiPixelPhase1TrackClusters
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1TrackClusters/interface/SiPixelPhase1TrackClusters.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <bitset>
#include <mutex>


namespace {
  constexpr int NCols = 416;
  struct Module {
    using Cols = std::array<unsigned short,416>;
    Cols left={{0}};
    Cols right={{0}};
  };
  
  std::mutex modules_lock;
  std::vector<Module> modules(1184);
  

  struct Dumper {
    bool go=false;
    ~Dumper() {
      int kk=0;
      if (go)
      for ( auto const & m : modules) {
       auto dump = [](auto const & l) { 
       return std::accumulate(std::next(l.begin()), l.end(),
                                    std::to_string(l[0]), // start with first element
                                    [](std::string a, int b) {
                                        return a + ',' + std::to_string(b);
                                    });
       };
       std::cout << dump(m.left) << std::endl;
       std::cout << dump(m.right) << std::endl;
       // fold
       std::array<int,52> left={{0}}, right={{0}};
       for (int i=0; i<NCols; i+=52) for(int j=0;j<52;++j)  { left[j]+=m.left[i+j];right[j]+=m.right[i+j];}
       std::cout << dump(left) << std::endl;
       std::cout << dump(right) << std::endl;
       int pl=0,pr=0,dl=0,dr=0; for(int j=2;j<50;j+=2) {dl+=left[j];dr+=right[j];pl+=left[j+1];pr+=right[j+1];}
       std::cout << "eff " << kk++ << ' ' << float(dl)/float(dr) << ' ' << float(pl)/float(pr) << std::endl;
      }
     }
  };
  Dumper dumper;
}

SiPixelPhase1TrackClusters::SiPixelPhase1TrackClusters(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig) 
{
  dumper.go=true;
  clustersToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(iConfig.getParameter<edm::InputTag>("clusters"));

  tracksToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));

  offlinePrimaryVerticesToken_ = consumes<reco::VertexCollection>(std::string("offlinePrimaryVertices"));

  applyVertexCut_=iConfig.getUntrackedParameter<bool>("VertexCut",true);
}

void SiPixelPhase1TrackClusters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // get geometry
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  assert(tracker.isValid());
  
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(offlinePrimaryVerticesToken_, vertices);

  if (applyVertexCut_ && (!vertices.isValid() || vertices->size() == 0)) return;

  //get the map
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken( tracksToken_, tracks);

  if ( !tracks.isValid() ) {
    edm::LogWarning("SiPixelPhase1TrackClusters")  << "track collection is not valid";
    return;
  }
  
  // get clusters
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> >  clusterColl;
  iEvent.getByToken( clustersToken_, clusterColl );

  if ( !clusterColl.isValid() ) {
    edm::LogWarning("SiPixelPhase1TrackClusters")  << "pixel cluster collection is not valid";
    return;
  }
  
  // we need to store some per-cluster data. Instead of a map, we use a vector,
  // exploiting the fact that all custers live in the DetSetVector and we can 
  // use the same indices to refer to them.
  // corr_charge is not strictly needed but cleaner to have it.
  std::vector<bool>  ontrack    (clusterColl->data().size(), false);
  std::vector<float> corr_charge(clusterColl->data().size(), -1.0f);
  std::vector<float> etatk(clusterColl->data().size(), -1.0f);


  for (auto const & track : *tracks) {

    if (applyVertexCut_ && (
       track.pt() < 0.75 || track.numberOfValidHits()<8 
       || std::abs( track.dxy(vertices->at(0).position()) ) > 5*track.dxyError())) continue;

    bool isBpixtrack = false, isFpixtrack = false, crossesPixVol=false;

    // find out whether track crosses pixel fiducial volume (for cosmic tracks)
    double d0 = track.d0(), dz = track.dz(); 
    if(std::abs(d0)<15 && std::abs(dz)<50) crossesPixVol = true;

    auto const & trajParams = track.extra()->trajParams();
    assert(trajParams.size()==track.recHitsSize());
    auto hb = track.recHitsBegin();
    for(unsigned int h=0;h<track.recHitsSize();h++){
      auto hit = *(hb+h);
      if (!hit->isValid()) continue;
      DetId id = hit->geographicalId();

      // check that we are in the pixel
      uint32_t subdetid = (id.subdetId());
      if (subdetid == PixelSubdetector::PixelBarrel) isBpixtrack = true;
      if (subdetid == PixelSubdetector::PixelEndcap) isFpixtrack = true;
      if (subdetid != PixelSubdetector::PixelBarrel && subdetid != PixelSubdetector::PixelEndcap) continue;
      auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
      if (!pixhit) continue;
        
      // get the cluster
      auto clust = pixhit->cluster();
      if (clust.isNull()) continue; 
      ontrack[clust.key()] = true; // mark cluster as ontrack
      

      // correct charge for track impact angle
      auto const & ltp = trajParams[h];
      LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
     
      float clust_alpha = atan2(localDir.z(), localDir.x());
      float clust_beta  = atan2(localDir.z(), localDir.y());
      double corrCharge = clust->charge() * sqrt( 1.0 / ( 1.0/pow( tan(clust_alpha), 2 ) + 
                                                          1.0/pow( tan(clust_beta ), 2 ) + 
                                                          1.0 ));
      /*
      if (subdetid == PixelSubdetector::PixelBarrel)
      std::cout << "corr charge " << clust->sizeY() << ' ' << clust->charge() << ' ' 
                << ltp.absdz() << ' ' << track.pt()/track.p() << ' ' <<  track.eta() << ' '
                << corrCharge << ' ' << clust->charge() *ltp.absdz() << std::endl;
      */
      corr_charge[clust.key()] = (float) corrCharge;
      etatk[clust.key()]=track.eta();
    }

    // statistics on tracks
    histo[NTRACKS].fill(1, DetId(0), &iEvent);
    if (isBpixtrack || isFpixtrack) 
      histo[NTRACKS].fill(2, DetId(0), &iEvent);
    if (isBpixtrack) 
      histo[NTRACKS].fill(3, DetId(0), &iEvent);
    if (isFpixtrack) 
      histo[NTRACKS].fill(4, DetId(0), &iEvent);

    if (crossesPixVol) {
      if (isBpixtrack || isFpixtrack)
        histo[NTRACKS_VOLUME].fill(1, DetId(0), &iEvent);
      else 
        histo[NTRACKS_VOLUME].fill(0, DetId(0), &iEvent);
    }
  }

  edmNew::DetSetVector<SiPixelCluster>::const_iterator it;
  for (it = clusterColl->begin(); it != clusterColl->end(); ++it) {
    auto id = DetId(it->detId());
    auto subdetid = (id.subdetId());
    const PixelGeomDetUnit* geomdetunit = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(id) );

    if( subdetid != PixelSubdetector::PixelBarrel) continue;
    auto seqnum = geomdetunit->index();

    std::bitset<416> cols;

    const PixelTopology& topol = geomdetunit->specificTopology();

    for(auto subit = it->begin(); subit != it->end(); ++subit) {
      // we could do subit-...->data().front() as well, but this seems cleaner.
      auto key = edmNew::makeRefTo(clusterColl, subit).key(); 
      bool is_ontrack = ontrack[key];
      if(!is_ontrack) continue;
      float corrected_charge = corr_charge[key];
      SiPixelCluster const& cluster = *subit;
      {
        auto const & off = cluster.pixelOffset();
        auto sz = off.size();
        auto mm = cluster.minPixelCol();
        for (auto i=1U; i<sz; i+=2) cols.set(mm+off[i]);
      }
      if (std::abs(etatk[key])<1.4f) continue;

      LocalPoint clustlp = topol.localPosition(MeasurementPoint(cluster.x(), cluster.y()));
      GlobalPoint clustgp = geomdetunit->surface().toGlobal(clustlp);
        
     if (cluster.minPixelCol()==0) continue;
     if	(cluster.maxPixelCol()+1==topol. ncolumns()) continue;

     // if ((cluster.minPixelCol()%topol.colsperroc()%2)==1) continue;

     auto sizeY = cluster.sizeY();
     // if (sizeY!=2) continue;
     if (topol.containsBigPixelInY(cluster.minPixelCol(), cluster.maxPixelCol()) ) continue;
      // sizeY+=1;

      //if (is_ontrack) {
      //if((cluster.minPixelCol()%topol.colsperroc()%2)==1) {
      if (sizeY<=2) {
        histo[ONTRACK_NCLUSTERS ].fill(id, &iEvent);
        histo[ONTRACK_CHARGE    ].fill(double(corrected_charge), id, &iEvent);
        histo[ONTRACK_SIZE      ].fill(double(cluster.size()  ), id, &iEvent);
        histo[ONTRACK_POSITION_B].fill(clustgp.z(),   clustgp.phi(),   id, &iEvent);
        histo[ONTRACK_POSITION_F].fill(clustgp.x(),   clustgp.y(),     id, &iEvent);
	histo[ONTRACK_SIZE_VS_ETA].fill(etatk[key], sizeY, id, &iEvent);
      } else if (sizeY>=4){
        histo[OFFTRACK_NCLUSTERS ].fill(id, &iEvent);
        histo[OFFTRACK_CHARGE    ].fill(double(cluster.charge()), id, &iEvent);
        histo[OFFTRACK_SIZE      ].fill(double(cluster.size()  ), id, &iEvent);
        histo[OFFTRACK_POSITION_B].fill(clustgp.z(),   clustgp.phi(),   id, &iEvent);
        histo[OFFTRACK_POSITION_F].fill(clustgp.x(),   clustgp.y(),     id, &iEvent);
      }
    }
    auto left = cols<<1;
    auto right = cols>>1;
    left &=cols;
    right &=cols;
    {
    std::lock_guard<std::mutex> guard(modules_lock);
    for (int i=0; i<NCols; ++i) {
      if (left[i]) ++modules[seqnum].left[i];
      if (right[i]) ++modules[seqnum].right[i];
    }
    }
  }

  histo[ONTRACK_NCLUSTERS].executePerEventHarvesting(&iEvent);
  histo[OFFTRACK_NCLUSTERS].executePerEventHarvesting(&iEvent);
}

DEFINE_FWK_MODULE(SiPixelPhase1TrackClusters);

