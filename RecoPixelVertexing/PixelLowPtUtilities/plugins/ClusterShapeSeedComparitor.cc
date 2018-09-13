#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"


#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseParams.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseDim.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseSigma.h"


#include <cstdio>
#include <cassert>

#include<iostream>

class PixelClusterShapeSeedComparitor : public SeedComparitor {
    public:
        PixelClusterShapeSeedComparitor(const edm::ParameterSet &cfg, edm::ConsumesCollector& iC) ;
        ~PixelClusterShapeSeedComparitor() override ; 
        void init(const edm::Event& ev, const edm::EventSetup& es) override ;
        bool compatible(const SeedingHitSet  &hits) const override { return true; }
        bool compatible(const TrajectoryStateOnSurface &,
                SeedingHitSet::ConstRecHitPointer hit) const override ;
        bool compatible(const SeedingHitSet  &hits, 
                const GlobalTrajectoryParameters &helixStateAtVertex,
                const FastHelix                  &helix) const override ;

    private:
        bool compatibleHit(const TrackingRecHit &hit, const GlobalVector &direction) const ;

        float dnnChi2(SiPixelRecHit const & recHit, GlobalVector gdir) const;

        std::string filterName_;
        edm::ESHandle<ClusterShapeHitFilter> filterHandle_;
        edm::EDGetTokenT<SiPixelClusterShapeCache> pixelClusterShapeCacheToken_;
        const SiPixelClusterShapeCache *pixelClusterShapeCache_;

        edm::ESHandle<TrackerTopology> tTopoHandle_;

        const bool filterAtHelixStage_;
        const bool filterPixelHits_, filterStripHits_;
        const bool useDNN_;
};


PixelClusterShapeSeedComparitor::PixelClusterShapeSeedComparitor(const edm::ParameterSet &cfg, edm::ConsumesCollector& iC) :
    filterName_(cfg.getParameter<std::string>("ClusterShapeHitFilterName")),
    pixelClusterShapeCache_(nullptr),
    filterAtHelixStage_(cfg.getParameter<bool>("FilterAtHelixStage")),
    filterPixelHits_(cfg.getParameter<bool>("FilterPixelHits")),
    filterStripHits_(cfg.getParameter<bool>("FilterStripHits")),
    useDNN_(false) // (filterPixelHits_)
{
  if(filterPixelHits_) {
    pixelClusterShapeCacheToken_ = iC.consumes<SiPixelClusterShapeCache>(cfg.getParameter<edm::InputTag>("ClusterShapeCacheSrc"));
  }
  std::cout << "PixelClusterShapeSeedComparitor using dnn " << (useDNN_ ? "YES" : "NO") << std::endl;  
}

PixelClusterShapeSeedComparitor::~PixelClusterShapeSeedComparitor() 
{
}

void
PixelClusterShapeSeedComparitor::init(const edm::Event& ev, const edm::EventSetup& es) {
    es.get<CkfComponentsRecord>().get(filterName_, filterHandle_);
    if(filterPixelHits_) {
      edm::Handle<SiPixelClusterShapeCache> hcache;
      ev.getByToken(pixelClusterShapeCacheToken_, hcache);
      pixelClusterShapeCache_ = hcache.product();
    }
    es.get<TrackerTopologyRcd>().get(tTopoHandle_);
 
}


bool
PixelClusterShapeSeedComparitor::compatible(const TrajectoryStateOnSurface &tsos,
                                            SeedingHitSet::ConstRecHitPointer hit) const
{
    if (filterAtHelixStage_) return true;
    assert(hit->isValid() && tsos.isValid());
    return compatibleHit(*hit, tsos.globalDirection());
}


float PixelClusterShapeSeedComparitor::dnnChi2(SiPixelRecHit const & recHit, GlobalVector gdir) const {

    ClusEllipseDim dnnD;
    ClusEllipseSigma dnnS;
 

   auto const & tkTpl = *tTopoHandle_;

       
   auto ldir = recHit.det()->toLocal(gdir);
   
   auto id = recHit.geographicalId();
 
   bool isBarrel = id.subdetId() == PixelSubdetector::PixelBarrel;

   float thickness = isBarrel ? 0.0285f : 0.029f;  // phase1
   constexpr float ipx = 1.f/0.01f; constexpr float ipy = 1.f/0.015f;  // phase1 pitch

   auto tkdx = ipx*thickness * ldir.x()/ldir.z();
   auto tkdy = ipy*thickness * ldir.y()/ldir.z();
   if( tkdy<0) { tkdx = -tkdx;}
   tkdy = std::abs(tkdy);

   ClusEllipseParams cep; cep.fill(recHit,tkTpl);
   if (cep.m_layer==0) return -1.f;
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
   auto zy = (pdy-tkdy)/psy;

   return zx*zx+zy*zy;

}

bool
PixelClusterShapeSeedComparitor::compatible(const SeedingHitSet  &hits, 
        const GlobalTrajectoryParameters &helixStateAtVertex,
        const FastHelix                  &helix) const
{ 
    if (!filterAtHelixStage_) return true;

    if(!helix.isValid() //check still if it's a straight line, which are OK
       && !helix.circle().isLine())//complain if it's not even a straight line
      edm::LogWarning("InvalidHelix") << "PixelClusterShapeSeedComparitor helix is not valid, result is bad";

    float xc = helix.circle().x0(), yc = helix.circle().y0();

    GlobalPoint  vertex = helixStateAtVertex.position();
    GlobalVector momvtx = helixStateAtVertex.momentum();
    float x0 = vertex.x(), y0 = vertex.y();
    int nh=0; float chi2=0;
    for (unsigned int i = 0, n = hits.size(); i < n; ++i) {
        auto const  & hit = *hits[i];
        GlobalPoint pos = hit.globalPosition();
        float x1 = pos.x(), y1 = pos.y(), dx1 = x1 - xc, dy1 = y1 - yc;

        // now figure out the proper tangent vector
        float perpx = -dy1, perpy = dx1;
        if (perpx * (x1-x0) + perpy * (y1 - y0) < 0) {
            perpy = -perpy; perpx = -perpx;
        }
       
        // now normalize (perpx, perpy, 1.0) to momentum (px, py, pz)
        float perp2 = perpx*perpx + perpy*perpy; 
        float pmom2 = momvtx.x()*momvtx.x() + momvtx.y()*momvtx.y(), momz2 = momvtx.z()*momvtx.z(), mom2 = pmom2 + momz2;
        float perpscale = sqrt(pmom2/mom2 / perp2), zscale = sqrt((1-pmom2/mom2));
        GlobalVector gdir(perpx*perpscale, perpy*perpscale, (momvtx.z() > 0 ? zscale : -zscale));

        auto const & recHit = reinterpret_cast<BaseTrackerRecHit const &>(hit); 
        if (useDNN_ && recHit.isPixel()) {
            auto const & pRecHit = reinterpret_cast<SiPixelRecHit const &>(recHit);
            auto lc = dnnChi2(pRecHit,gdir);
            if (lc>=0) {
              ++nh; chi2+=lc;
            }
        } 
        else if (!compatibleHit(hit, gdir)) {
            return false; // not yet
        }
    }
    // if (useDNN_) std::cout << "PixelClusterShapeSeedComparitor chi2 " << chi2 << ' ' << nh << std::endl;
    if (useDNN_) return nh==0 || chi2 < 18.f*float(nh);
    return true; 
}

bool 
PixelClusterShapeSeedComparitor::compatibleHit(const TrackingRecHit &hit, const GlobalVector &direction) const 
{
    if (hit.geographicalId().subdetId() <= 2) {
        if (!filterPixelHits_) return true;    
        const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit *>(&hit);
        if (pixhit == nullptr) throw cms::Exception("LogicError", "Found a valid hit on the pixel detector which is not a SiPixelRecHit\n");
        //printf("Cheching hi hit on detid %10d, local direction is x = %9.6f, y = %9.6f, z = %9.6f\n", hit.geographicalId().rawId(), direction.x(), direction.y(), direction.z());
        return filterHandle_->isCompatible(*pixhit, direction, *pixelClusterShapeCache_);
    } else {
        if (!filterStripHits_) return true;
        const std::type_info &tid = typeid(*&hit);
        if (tid == typeid(SiStripMatchedRecHit2D)) {
            const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D *>(&hit);
            assert(matchedHit != nullptr);
            return (filterHandle_->isCompatible(DetId(matchedHit->monoId()), matchedHit->monoCluster(), direction) &&
                    filterHandle_->isCompatible(DetId(matchedHit->stereoId()), matchedHit->stereoCluster(), direction));
        } else if (tid == typeid(SiStripRecHit2D)) {
            const SiStripRecHit2D* recHit = dynamic_cast<const SiStripRecHit2D *>(&hit);
            assert(recHit != nullptr);
            return filterHandle_->isCompatible(*recHit, direction);
        } else if (tid == typeid(ProjectedSiStripRecHit2D)) {
            const ProjectedSiStripRecHit2D* precHit = dynamic_cast<const ProjectedSiStripRecHit2D *>(&hit);
            assert(precHit != nullptr);
            return filterHandle_->isCompatible(precHit->originalHit(), direction);
        } else {
            //printf("Questo e' un %s, che ci fo?\n", tid.name());
            return true;
        }
    }
}

DEFINE_EDM_PLUGIN(SeedComparitorFactory, PixelClusterShapeSeedComparitor, "PixelClusterShapeSeedComparitor");
