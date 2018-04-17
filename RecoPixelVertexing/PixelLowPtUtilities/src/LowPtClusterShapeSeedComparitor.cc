#include "RecoPixelVertexing/PixelLowPtUtilities/interface/LowPtClusterShapeSeedComparitor.h"

#include "RecoPixelVertexing/PixelTrackFitting/src/CircleFromThreePoints.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"


#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseParams.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseDim.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseSigma.h"


#include<cmath>



namespace {
  typedef Basic2DVector<float>   Vector2D;

  inline float sqr(float x) { return x*x; }

  /*****************************************************************************/
  inline
  float areaParallelogram
  (const Vector2D& a, const Vector2D& b)
  {  
    return a.x() * b.y() - a.y() * b.x();
  }
  
  /*****************************************************************************/

  inline 
  bool getGlobalDirs(GlobalPoint const * g, GlobalVector * globalDirs)
  {
    
    
    // Determine circle
    CircleFromThreePoints circle(g[0],g[1],g[2]);
    
    float curvature = circle.curvature();
    if(0.f == curvature) {
      LogDebug("LowPtClusterShapeSeedComparitor")<<"the curvature is null:"
						 <<"\n point1: "<<g[0]
						 <<"\n point2: "<<g[1]
						 <<"\n point3: "<<g[2];
      return false;
    }

   // Get 2d points
    Vector2D p[3];
    Vector2D c  = circle.center();
    for(int i=0; i!=3; i++)
      p[i] =  g[i].basicVector().xy() -c;
 

    float area = std::abs(areaParallelogram(p[1] - p[0], p[1]));
    
    float a12 = std::asin(std::min(area*curvature*curvature,1.f));
    
    float slope = (g[1].z() - g[0].z()) / a12;
 
    // Calculate globalDirs
   
    float cotTheta = slope * curvature; 
    float sinTheta = 1.f/std::sqrt(1.f + sqr(cotTheta));
    float cosTheta = cotTheta*sinTheta;
    
    if (areaParallelogram(p[0], p[1] ) < 0)  sinTheta = - sinTheta;
        
    for(int i = 0; i!=3;  i++) {
      Vector2D vl = p[i]*(curvature*sinTheta);
      globalDirs[i] = GlobalVector(-vl.y(),
				    vl.x(),
				    cosTheta
				    );
    }
    return true;
  }

  /*****************************************************************************/

  
  inline
  void getGlobalPos(const SeedingHitSet &hits, GlobalPoint * globalPoss)
  {
    
    for(unsigned int i=0; i!=hits.size(); ++i)
  	globalPoss[i] = hits[i]->globalPosition();
  }

} // namespace

/*****************************************************************************/
LowPtClusterShapeSeedComparitor::LowPtClusterShapeSeedComparitor(const edm::ParameterSet& ps, edm::ConsumesCollector& iC):
  thePixelClusterShapeCacheToken(iC.consumes<SiPixelClusterShapeCache>(ps.getParameter<edm::InputTag>("clusterShapeCacheSrc"))),
  theShapeFilterLabel_(ps.getParameter<std::string>("clusterShapeHitFilter"))
{}

/*****************************************************************************/
void LowPtClusterShapeSeedComparitor::init(const edm::Event& e, const edm::EventSetup& es) {
  es.get<CkfComponentsRecord>().get(theShapeFilterLabel_, theShapeFilter);
  es.get<TrackerTopologyRcd>().get(theTTopo);

  e.getByToken(thePixelClusterShapeCacheToken, thePixelClusterShapeCache);


}

constexpr bool useDNN = true;
float dnnChi2(SiPixelRecHit const & recHit, GlobalVector gdir, TrackerTopology const & tkTpl) {

    ClusEllipseDim dnnD;
    ClusEllipseSigma dnnS;
 
    constexpr bool extendedModel = true;

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
   if(extendedModel) {
     for (int kk = 9; kk<16; ++kk) dnnS.arg0_data()[kk] = dnnD.arg0_data()[kk]=0;
     int ind = dnnS.arg0_data()[1] + 3.f*dnnS.arg0_data()[0] -1.f;
     assert(ind<8);
     dnnS.arg0_data()[9+ind] = dnnD.arg0_data()[9+ind] = 1;
   }

   dnnD.Run();
   dnnS.Run();

   tkdx = cep.m_sy>1 ? tkdx : std::abs(tkdx);

   auto pdx = 0.25f *(cep.m_dx + dnnD.result0_data()[0]);
   auto pdy = cep.m_dy + dnnD.result0_data()[1];

   auto psx = dnnS.result0_data()[0];
   auto psy = dnnS.result0_data()[1];

   auto zx = (pdx-tkdx)/psx;
   auto zy = (pdy-tkdy)/psy;


   /* if trained with MB 0PU with extended DNN */
   // in endcap the x-distribution has peaks at +/-1 :  try to correct
   if (!isBarrel  && cep.m_sy<3.f ) zx = 1.5f*std::max(0.f,std::abs(zx)-1.f);
   // in endcap the y-distribution is shifted toward negative values for large size...
   if (!isBarrel && cep.m_sy==3.f) zy -=0.8f;
   if (!isBarrel && cep.m_sy>3.f) zy +=0.8f;
   
   // in Barrel L1 there is a tail for large negative zy at large PU due to ??Broken clusters??  
   if (isBarrel&&cep.m_layer==1.f&& cep.m_sy>4.f) zy = zy>-2.f ? zy : 0.75f*zy; // zy = std::max(-3.5f,zy);
   


   /* if trained with MB 0PU
   // in endcap the x-distribution has peaks at +/-1 :  try to correct
   if (!isBarrel) zx = 1.5f*std::max(0.f,std::abs(zx)-1.f);
   // in endcap the y-distribution is shifted toward negative values for large size...
   if (!isBarrel && cep.m_sy>3) zy +=0.5f;

   // in Barrel L1 there is a tail for large negative zy at large PU due to DynIneff
   // if (isBarrel&&cep.m_layer==1.f) zy = zy>0 ? zy : 0.5f*zy; // zy = std::max(-3.5f,zy);
   */

   /* if trained with realisitc ttbar 50PU
   // in endcap the x-distribution has peaks at +/-1 :  try to correct
   if (!isBarrel) zx = 1.5f*std::max(0.f,std::abs(zx)-1.f);
   // in endcap the y-distribution is shifted toward negative values for large size ...
   if (!isBarrel && cep.m_sy>3) zy +=0.75f;
   // in Barrel L1 is sharply peaked at +1 with a negative tail...
   if (isBarrel&&cep.m_layer==1.f) {zy-=0.8f; zy = zy>0 ? 2.f*zy : zy;}
   */

   if (std::max(std::abs(zx),std::abs(zy))>5.f) return 100.f; // kill outliers
   auto chi2 = zx*zx+zy*zy;
   return chi2;

}




bool LowPtClusterShapeSeedComparitor::compatible(const SeedingHitSet &hits) const
//(const reco::Track* track, const vector<const TrackingRecHit *> & recHits) const
{
  assert(hits.size()==3);

  const ClusterShapeHitFilter * filter = theShapeFilter.product();
  if(filter == nullptr)
    throw cms::Exception("LogicError") << "LowPtClusterShapeSeedComparitor: init(EventSetup) method was not called";

   // Get global positions
   GlobalPoint  globalPoss[3];
   getGlobalPos(hits, globalPoss);

  // Get global directions
  GlobalVector globalDirs[3]; 

  bool ok = getGlobalDirs(globalPoss,globalDirs);

  // Check whether shape of pixel cluster is compatible
  // with local track direction

  if (!ok)
    {
      LogDebug("LowPtClusterShapeSeedComparitor")<<"curvarture 0:"
						 <<"\nnHits: "<<hits.size()
						 <<" will say the seed is good anyway.";
      return true;
    }

  int nh=0; float chi2=0;
 
  for(int i = 0; i < 3; i++)
  {
    const SiPixelRecHit* pixelRecHit =
      dynamic_cast<const SiPixelRecHit *>(hits[i]->hit());

    if (!pixelRecHit){
      edm::LogError("LowPtClusterShapeSeedComparitor")<<"this is not a pixel cluster";
      ok = false; break;
    }

    if(!pixelRecHit->isValid())
    { 
      ok = false; break; 
    }
    
    LogDebug("LowPtClusterShapeSeedComparitor")<<"about to compute compatibility."
					       <<"hit ptr: "<<pixelRecHit
					       <<"global direction:"<< globalDirs[i];


    if (useDNN) {
      auto lc = dnnChi2(*pixelRecHit,globalDirs[i],*theTTopo);
      if (lc>=0) {
        // if (lc>32.f) return false; // kill outliers...
        ++nh; chi2+=lc;
      }
    }else
    if(! filter->isCompatible(*pixelRecHit, globalDirs[i], *thePixelClusterShapeCache) )
    {
      LogTrace("LowPtClusterShapeSeedComparitor")
         << " clusShape is not compatible"
         << HitInfo::getInfo(*hits[i]->hit(),theTTopo.product());

      ok = false; break;
    }
  }
  // if (useDNN) std::cout << "LowPtClusterShapeSeedComparitor chi2 " << chi2 << ' ' << nh << std::endl;
  if (useDNN) return nh==0 || chi2 < 24.f*float(nh);
 
  return ok;
}

