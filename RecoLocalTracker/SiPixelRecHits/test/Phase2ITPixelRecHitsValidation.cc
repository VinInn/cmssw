#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"


#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include <TH1D.h>
#include <TH2D.h>

typedef TH1F* TH1Fptr;
typedef TH2F* TH2Fptr;

class Phase2ITPixelRecHitsValidation : public edm::one::EDAnalyzer<> {

public:
  
  explicit Phase2ITPixelRecHitsValidation(const edm::ParameterSet&);
  ~Phase2ITPixelRecHitsValidation();
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  
private:
    
  edm::EDGetTokenT< SiPixelRecHitCollectionNew > tokenRecHits_;        

  // histograms
  edm::Service<TFileService> fs_;

  //  TH1D* numberRecHitsPixel;
  //  TH1D* numberRecHitsStrip;
  
  TH2F* trackerLayoutRz_;
  TH2F* trackerLayoutXY_;
  TH2F* trackerLayoutXYBPIX_;
  TH2F* trackerLayoutXYFPIX_[2];
  
  TFileDirectory BPIXdir_;
  TFileDirectory FPIXnegdir_;
  TFileDirectory FPIXposdir_;
  
  const int MAX_BPIX=4;
  const int MAX_FPIX=10;

  // hit maps
  TH2Fptr * h2_yxLocalHitMap_BPIX_;
  TH2Fptr * h2_yxLocalHitMap_FPIXneg_;
  TH2Fptr * h2_yxLocalHitMap_FPIXpos_;

};

Phase2ITPixelRecHitsValidation::Phase2ITPixelRecHitsValidation(const edm::ParameterSet& conf) :
  tokenRecHits_(consumes< SiPixelRecHitCollectionNew >(conf.getParameter<edm::InputTag>("phase2ITPixelRecHitSrc")))
{
  
  LogDebug("Phase2ITPixelRecHitsValidation") << " C'TOR called" << std::endl;

}

Phase2ITPixelRecHitsValidation::~Phase2ITPixelRecHitsValidation() { 

  if ( h2_yxLocalHitMap_BPIX_ )    delete[] h2_yxLocalHitMap_BPIX_ ;  
  if ( h2_yxLocalHitMap_FPIXneg_ ) delete[] h2_yxLocalHitMap_FPIXneg_;  
  if ( h2_yxLocalHitMap_FPIXpos_ ) delete[] h2_yxLocalHitMap_FPIXpos_;  

}

void Phase2ITPixelRecHitsValidation::beginJob() {
  edm::Service<TFileService> fs_; 
  
  TFileDirectory td_ = fs_->mkdir("Common");
  // Create common histograms
  trackerLayoutRz_      = td_.make< TH2F >("RVsZ", "R vs. z position; z [cm]; R [cm]",    6000,-300.0, 300.0, 300, 0.0, 30.0);
  trackerLayoutXY_      = td_.make< TH2F >("XVsY", "x vs. y position; x [cm]; y [cm]",     600, -30.0,  30.0, 600, -30.0, 30.0);

  // hit maps 
  h2_yxLocalHitMap_BPIX_    = new TH2Fptr[MAX_BPIX];
  h2_yxLocalHitMap_FPIXneg_ = new TH2Fptr[MAX_FPIX];
  h2_yxLocalHitMap_FPIXpos_ = new TH2Fptr[MAX_FPIX];

  char buff_name[60];
  char buff_title[60];
  
  // BPIX
  BPIXdir_ = fs_->mkdir( "BPIX" );
  trackerLayoutXYBPIX_ = BPIXdir_.make< TH2F >("XVsYBPIX", "x vs. y position; x [cm]; y [cm] ",600, -30.0,  30.0, 600, -30.0, 30.0);
  for ( int i=0; i<MAX_BPIX; i++ ) {
    sprintf(buff_name, "h2_yxLocalHitMap_BPIX_%u",i+1);
    sprintf(buff_title,"h2_yxLocalHitMap_BPIX_%u; y[cm] ; x[cm]",i+1);    
    h2_yxLocalHitMap_BPIX_[i] = BPIXdir_.make<TH2F>( buff_name,buff_title,600,-3.,+3.,100,-1.,+1.);

  }  

  // FPIX neg
  FPIXnegdir_ = fs_->mkdir( "FPIXneg" );
  trackerLayoutXYFPIX_[0]  = FPIXnegdir_.make< TH2F >("XVsYFPIXneg", "x vs. y position; x [cm]; y [cm]", 600, -30.0, 30.0, 600, -30.0, 30.0);
  for ( int i=0; i<MAX_FPIX; i++ ) {
    sprintf(buff_name, "h2_yxLocalHitMap_FPIXneg_%u",i+1);
    sprintf(buff_title,"h2_yxLocalHitMap_FPIXneg_%u; y [cm]; x [cm]",i+1);    
    h2_yxLocalHitMap_FPIXneg_[i] = FPIXnegdir_.make<TH2F>( buff_name,buff_title,600,-3.,+3.,100,-1.,+1.);
  }

  // FPIX pos
  FPIXposdir_ = fs_->mkdir( "FPIXpos" );
  trackerLayoutXYFPIX_[1] = FPIXposdir_.make< TH2F >("XVsYFPIXpos", "x vs. y position; x [cm]; y [cm]", 600, -30.0, 30.0, 600, -30.0, 30.0);
  for ( int i=0; i<MAX_FPIX; i++ ) {
    sprintf(buff_name, "h2_yxLocalHitMap_FPIXpos_%u",i+1);
    sprintf(buff_title,"h2_yxLocalHitMap_FPIXpos_%u; y [cm]; x [cm]",i+1);    
    h2_yxLocalHitMap_FPIXpos_[i] = FPIXposdir_.make<TH2F>( buff_name,buff_title,600,-3.,+3.,100,-1.,+1.);
  }
  
}


void Phase2ITPixelRecHitsValidation::endJob() { }

void Phase2ITPixelRecHitsValidation::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {

    /*
     * Get the needed objects
     */

    // Get the RecHits
    edm::Handle< SiPixelRecHitCollectionNew > rechits;
    event.getByToken(tokenRecHits_, rechits);

    // Get the geometry
    edm::ESHandle< TrackerGeometry > tGeomHandle;
    eventSetup.get< TrackerDigiGeometryRecord >().get(tGeomHandle);
    const TrackerGeometry* trackerGeometry = &(*tGeomHandle);

    edm::ESHandle< TrackerTopology > tTopoHandle;
    eventSetup.get< TrackerTopologyRcd >().get(tTopoHandle);
    const TrackerTopology* trackerTopology = tTopoHandle.product();

    /*
     * Validation: hit maps
     */

    // Loop over modules
    for (SiPixelRecHitCollectionNew::const_iterator DSViter = rechits->begin(); DSViter != rechits->end(); ++DSViter) {
      
      // Get the detector unit's id
      unsigned int rawid(DSViter->detId()); 
      DetId detId(rawid);
      
      LogDebug("Phase2ITPixelClusterRecHitsValidation") << " LOOP on detId detId/trackerGeometry->idToDetUnit(detId) " << rawid << " " << trackerGeometry->idToDetUnit(detId) << std::endl;

      // Get the geometry of the tracker
      const GeomDetUnit* geomDetUnit(trackerGeometry->idToDetUnit(detId));
      //      const PixelGeomDetUnit* theGeomDet = dynamic_cast< const PixelGeomDetUnit* >(geomDetUnit);
      //      const PixelTopology& topol = theGeomDet->specificTopology();
      
      if (!geomDetUnit) break;
      
      // Number of clusters
      //      unsigned int nRecHitsPixel(0), nRecHitsStrip(0);
      
      // Loop over the clusters in the detector unit
      for (edmNew::DetSet< SiPixelRecHit >::const_iterator rechitIt = DSViter->begin(); rechitIt != DSViter->end(); ++rechitIt) {
	
	/*
	 * Cluster related variables
	 */
	  
	LocalPoint localPosClu = rechitIt->localPosition();
	Global3DPoint globalPosClu = geomDetUnit->surface().toGlobal(localPosClu);
	
	// Fill the position histograms
	trackerLayoutRz_->Fill(globalPosClu.z(), globalPosClu.perp());
	trackerLayoutXY_->Fill(globalPosClu.x(), globalPosClu.y());

	//###

	if ( detId.subdetId() == PixelSubdetector::PixelBarrel) {
	  // BPIX
	  trackerLayoutXYBPIX_->Fill(globalPosClu.x(), globalPosClu.y());
	  int layer = trackerTopology->pxbLayer(detId);
	  h2_yxLocalHitMap_BPIX_[layer-1]->Fill(localPosClu.y(),localPosClu.x());
	} else if (detId.subdetId() == PixelSubdetector::PixelEndcap) {
	  int layer = trackerTopology->pxfDisk(detId);
	  if ( trackerTopology->pxfSide(detId) == 1 ) { 
	    // FPIX z<0
	    trackerLayoutXYFPIX_[0]->Fill(globalPosClu.x(), globalPosClu.y());
	    h2_yxLocalHitMap_FPIXneg_[layer-1]->Fill(localPosClu.y(),localPosClu.x());
	  } else if ( trackerTopology->pxfSide(detId) == 2 ) { 
	    // FPIX z>0
	    trackerLayoutXYFPIX_[1]->Fill(globalPosClu.x(), globalPosClu.y());
	    h2_yxLocalHitMap_FPIXpos_[layer-1]->Fill(localPosClu.y(),localPosClu.x());
	  }
	}
      } // end of loop on RecHits
    } // end of loop od DetSet

}


DEFINE_FWK_MODULE(Phase2ITPixelRecHitsValidation);
