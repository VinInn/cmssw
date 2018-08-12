#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <RecoTracker/MeasurementDet/interface/MeasurementTracker.h>
#include <RecoTracker/Record/interface/CkfComponentsRecord.h>
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"


#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

namespace {

Surface::RotationType rotation( const GlobalVector& zDir)
{
  GlobalVector zAxis = zDir.unit();
  GlobalVector yAxis( zAxis.y(), -zAxis.x(), 0); 
  GlobalVector xAxis = yAxis.cross( zAxis);
  return Surface::RotationType( xAxis, yAxis, zAxis);
}


}

class Det2DetPropTest : public edm::EDAnalyzer {
public:
  explicit Det2DetPropTest(const edm::ParameterSet&);
  ~Det2DetPropTest();


private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::string theMeasurementTrackerName;
  std::string theNavigationSchoolName;
};


Det2DetPropTest::Det2DetPropTest(const edm::ParameterSet& iConfig): 
   theMeasurementTrackerName(iConfig.getParameter<std::string>("measurementTracker"))
  ,theNavigationSchoolName(iConfig.getParameter<std::string>("navigationSchool")){}

Det2DetPropTest::~Det2DetPropTest() {}

void Det2DetPropTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  //get the measurementtracker
  edm::ESHandle<MeasurementTracker> measurementTracker;
  edm::ESHandle<NavigationSchool>   navSchool;

  iSetup.get<CkfComponentsRecord>().get(theMeasurementTrackerName, measurementTracker);
  iSetup.get<NavigationSchoolRecord>().get(theNavigationSchoolName, navSchool);

  auto const & geom = *(TrackerGeometry const *)(*measurementTracker).geomTracker();
  auto const & dus = geom.detUnits();
  
  auto firstBarrel = geom.offsetDU(GeomDetEnumerators::tkDetEnum[1]);
  auto firstForward = geom.offsetDU(GeomDetEnumerators::tkDetEnum[2]);
 
  std::cout << "number of dets " << dus.size() << std::endl;
  std::cout << "Bl/Fw loc " << firstBarrel<< '/' << firstForward << std::endl;

  edm::ESHandle<MagneticField> magfield;
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);

  edm::ESHandle<Propagator>             propagatorHandle;
  iSetup.get<TrackingComponentsRecord>().get("RungeKuttaTrackerPropagator", propagatorHandle);
  // iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", propagatorHandle);
  auto & prop = *(*propagatorHandle).clone();  // leak

  // error (very very small)
  ROOT::Math::SMatrixIdentity id;
  AlgebraicSymMatrix55 C(id);
  C *= 1.e-16;

  CurvilinearTrajectoryError err(C);

  
  KFUpdator kfu;
  LocalError he(0.01*0.01,0,0.02*0.02);



  auto const & detI = *geom.idToDet(352459020);
  auto const & detO = *geom.idToDet(402676364);

  GlobalVector vdir(detO.position() - detI.position());
  float p = 45.0f;
  GlobalVector startingMomentum  = p*vdir/vdir.mag();

  std::cout << "Angles " << startingMomentum.phi()<< ' ' << startingMomentum.eta()  << std::endl;

  GlobalPoint startingPosition = detI.position();
  GlobalPoint endingPosition = detO.position();

  // make TSOS happy
  //Define starting plane

  auto const & startingPlane = detI.surface();                    
  auto const & endingPlane = detO.surface();

  auto overP = 1.f/startingMomentum.mag();
  auto dop = 0.01f*overP;
  GlobalVector moms[3] = { overP/(overP-dop)*startingMomentum , startingMomentum,  overP/(overP+dop)*startingMomentum};

  
  // GlobalVector moms[3] = { 0.5f*startingMomentum,startingMomentum,10.f*startingMomentum};
  
  LocalPoint inner[6],outer[6];
  int n=0;
  for (int charge=-1;charge<=1;charge+=2)
  for (auto mom : moms) {
    prop.setPropagationDirection(alongMomentum);
    std::cout << "\n\nMom " << charge << ' ' << mom << ' ' << mom.perp() << std::endl;

    TrajectoryStateOnSurface startingStateP( GlobalTrajectoryParameters(startingPosition, 
	  				          mom, charge, magfield.product()), 
				                  err, startingPlane);
    auto tsos = startingStateP;

    std::cout << "Start Mon " << tsos.globalMomentum() << ' ' << tsos.globalMomentum().perp() << std::endl;
    std::cout << "Start Pos " << tsos.globalPosition() << ' ' << tsos.localPosition() << ' ' << tsos.localError().positionError() << std::endl;

    auto  endStateP = prop.propagate( tsos, endingPlane);
  
    tsos = endStateP;

    std::cout << "End Mon " << tsos.globalMomentum() << ' ' << tsos.globalMomentum().perp() << std::endl;
    std::cout << "End Pos " << tsos.globalPosition() << ' ' << tsos.localPosition() << ' ' << tsos.localError().positionError() << std::endl; 

    outer[n] = tsos.localPosition();

    std::cout << "\nNOW Backward" << std::endl;

    prop.setPropagationDirection(oppositeToMomentum);

    TrajectoryStateOnSurface startingStateE( GlobalTrajectoryParameters(endingPosition,
                                                  mom, charge, magfield.product()),
                                                  err, endingPlane);


    tsos = startingStateE;

    std::cout << "Start Mon " << tsos.globalMomentum() << ' ' << tsos.globalMomentum().perp() << std::endl;
    std::cout << "Start Pos " << tsos.globalPosition() << ' ' << tsos.localPosition() << ' ' << tsos.localError().positionError() << std::endl;

    auto  endStateE = prop.propagate( tsos, startingPlane);

    tsos = endStateE;

    std::cout << "End Mon " << tsos.globalMomentum() << ' ' << tsos.globalMomentum().perp() << std::endl;
    std::cout << "End Pos " << tsos.globalPosition() << ' ' << tsos.localPosition() << ' ' << tsos.localError().positionError() << std::endl;

    inner[n] = tsos.localPosition();
    ++n;
  } // end loop mon & charge

  constexpr float toum = 10000.;
  std::cout << "\nouter" << std::endl;
  std::cout << toum*(outer[0]-outer[1]) << std::endl;
  std::cout << toum*(outer[2]-outer[1]) << std::endl;
  std::cout << toum*(outer[3]-outer[4]) << std::endl;
  std::cout << toum*(outer[5]-outer[4]) << std::endl;

  std::cout << "\ninner" << std::endl;
  std::cout << toum*(inner[0]-inner[1]) << std::endl;
  std::cout << toum*(inner[2]-inner[1]) << std::endl;
  std::cout << toum*(inner[3]-inner[4]) << std::endl;
  std::cout << toum*(inner[5]-inner[4]) << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(Det2DetPropTest);
