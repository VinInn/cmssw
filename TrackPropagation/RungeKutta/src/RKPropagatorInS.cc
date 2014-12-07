#include "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"
#include "RKCartesianDistance.h"
#include "CartesianLorentzForce.h"
#include "RKLocalFieldProvider.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "RKAdaptiveSolver.h"
#include "RKOne4OrderStep.h"
#include "RKOneCashKarpStep.h"
#include "PathToPlane2Order.h"
#include "CartesianStateAdaptor.h"
#include "TrackingTools/GeomPropagators/interface/StraightLineCylinderCrossing.h"
#include "TrackingTools/GeomPropagators/interface/StraightLineBarrelCylinderCrossing.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "TrackingTools/GeomPropagators/interface/PropagationDirectionFromPath.h"
#ifdef EDM_LM_DEBUG
#include "frameChanger.h"
#endif
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "AnalyticalErrorPropagation.h"
#include "GlobalParametersWithPath.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Likely.h"


std::pair<TrajectoryStateOnSurface,double>
RKPropagatorInS::propagateWithPath(const FreeTrajectoryState& fts, 
				   const Plane& plane) const
{
  GlobalParametersWithPath gp =  propagateParametersOnPlane( fts, plane.hessianPlaneDouble());
  if unlikely(!gp) return TsosWP(TrajectoryStateOnSurface(),0.);

  SurfaceSideDefinition::SurfaceSide side = PropagationDirectionFromPath()(gp.s(),propagationDirection())==alongMomentum 
    ? SurfaceSideDefinition::beforeSurface : SurfaceSideDefinition::afterSurface;
  return  analyticalErrorPropagation( fts, plane, side, gp.parameters(),gp.s());
}

std::pair< TrajectoryStateOnSurface, double> 
RKPropagatorInS::propagateWithPath (const FreeTrajectoryState& fts, const Cylinder& cyl) const
{
  GlobalParametersWithPath gp =  propagateParametersOnCylinder( fts, cyl);
  if  unlikely(!gp) return TsosWP(TrajectoryStateOnSurface(),0.);

  SurfaceSideDefinition::SurfaceSide side = PropagationDirectionFromPath()(gp.s(),propagationDirection())==alongMomentum 
    ? SurfaceSideDefinition::beforeSurface : SurfaceSideDefinition::afterSurface;
  return analyticalErrorPropagation( fts, cyl, side, gp.parameters(),gp.s());
  
}

GlobalParametersWithPath
RKPropagatorInS::propagateParametersOnPlane( const FreeTrajectoryState& ts, 
					     const HessianPlane<double>& plane) const
{

  GlobalPoint gpos( ts.position());
  GlobalVector gmom( ts.momentum());
  double startZ = plane.localZ(gpos.basicVector());
  // (transverse) curvature
  double rho = ts.transverseCurvature();
  //
  // Straight line approximation? |rho|<1.e-10 equivalent to ~ 1um 
  // difference in transversal position at 10m.
  //
  if  unlikely( fabs(rho)<1.e-10 ) {
    //
    // Instantiate auxiliary object for finding intersection.
    // Frame-independant point and vector are created explicitely to 
    // avoid confusing gcc (refuses to compile with temporary objects
    // in the constructor).
    //
    LogDebug("RKPropagatorInS")<<" startZ = "<<startZ;

    if unlikely(fabs(startZ) < 1e-5){  
      LogDebug("RKPropagatorInS")<< "Propagation is not performed: state is already on final surface.";
      GlobalTrajectoryParameters res( gpos, gmom, ts.charge(), 
				      &theVolume);
      return GlobalParametersWithPath( res, 0.0);
    }

    StraightLinePlaneCrossing::PositionType pos(gpos);
    StraightLinePlaneCrossing::DirectionType dir(gmom);
    StraightLinePlaneCrossing planeCrossing(pos,dir, propagationDirection());
    //
    // get solution
    //
    std::pair<bool,double> propResult = planeCrossing.pathLength(plane);
    if likely( propResult.first) {
      double s = propResult.second;
      // point (reconverted to GlobalPoint)
      GlobalPoint x (planeCrossing.position(s));
      GlobalTrajectoryParameters res( x, gmom, ts.charge(), &theVolume);
      return GlobalParametersWithPath( res, s);
      }
    //do someting
    LogDebug("RKPropagatorInS")<< "Straight line propgation to plane failed !!"; 
    return GlobalParametersWithPath( );

  }


#ifdef EDM_LM_DEBUG
    LogDebug("RKPropagatorInS")  << "RKPropagatorInS: starting prop to plane in volume with pos " << theVolume->position()
	      << " Z axis " << theVolume->toGlobal( LocalVector(0,0,1)) ;

    LogDebug("RKPropagatorInS")  << "The starting position is " << ts.position() << " (global) "
	      << theVolume->toLocal(ts.position()) << " (local) " ;
  
    auto localPlane = frameChanger::transform( plane, *theVolume);
    //LogDebug("RKPropagatorInS")  << "The plane position is " << plane.position() << " (global) "
   //	      << localPlane->position() << " (local) " ;

    LogDebug("RKPropagatorInS")  << "The initial distance to plane is " << plane.localZ( ts.position().basicVector()) ;

    StraightLinePlaneCrossing cross( ts.position().basicVector(), ts.momentum().basicVector());
    std::pair<bool,double> res3 = cross.pathLength(plane);
    LogDebug("RKPropagatorInS")  << "straight line distance " << res3.first << " " << res3.second ;
#endif

  typedef RKAdaptiveSolver<double,RKOneCashKarpStep, 6>   Solver;
  typedef Solver::Vector                                  RKVector;
  
  RKLocalFieldProvider field(theVolume);
  PathToPlane2Order pathLength(theVolume);
  CartesianLorentzForce deriv(field, ts.charge());

  RKCartesianDistance dist;
  double eps = theTolerance;
  Solver solver;
  double stot = 0;
  PropagationDirection currentDirection = propagationDirection();

  // in magVolume frame
  RKVector start( CartesianStateAdaptor::rkstate( rkPosition(gpos), rkMomentum(gmom)));
  int safeGuard = 0;
  while (safeGuard++<100) {
    CartesianStateAdaptor startState(start);

    std::pair<bool,double> path = pathLength( plane, startState.position(), 
					      startState.momentum(), 
					      ts.charge(), currentDirection);
    if  unlikely(!path.first) { 
      LogDebug("RKPropagatorInS")  << "RKPropagatorInS: Path length calculation to plane failed!" 
				   << "...distance to plane " << plane.localZ( globalPosition(startState.position()).basicVector())
				   << "...Local starting position in volume " << startState.position() 
				   << "...Magnetic field " << field.inTesla( startState.position()) ;


      return GlobalParametersWithPath();
    }

    LogDebug("RKPropagatorInS")  << "RKPropagatorInS: Path lenght to plane is " << path.second ;
   

    double sstep = path.second;
    if  unlikely( std::abs(sstep) < eps) {
      LogDebug("RKPropagatorInS")  << "On-surface accuracy not reached, but pathLength calculation says we are there! "
		 << "path " << path.second << " distance to plane is " << startZ ;
      GlobalTrajectoryParameters res( gtpFromVolumeLocal( startState, ts.charge()));
      return GlobalParametersWithPath( res, stot);
    }

    LogDebug("RKPropagatorInS")  << "RKPropagatorInS: Solving for " << sstep 
	      << " current distance to plane is " << startZ ;

    RKVector rkresult = solver( 0, start, sstep, deriv, dist, eps);
    stot += sstep;
    CartesianStateAdaptor cur( rkresult);
    double remainingZ = plane.localZ( globalPosition(cur.position()).basicVector());

    if ( fabs(remainingZ) < eps) {
      LogDebug("RKPropagatorInS")  << "On-surface accuracy reached! " << remainingZ ;
      GlobalTrajectoryParameters res( gtpFromVolumeLocal( cur, ts.charge()));
      return GlobalParametersWithPath( res, stot);
    }

    start = rkresult;

    if (remainingZ * startZ > 0) {
      LogDebug("RKPropagatorInS")  << "Accuracy not reached yet, trying in same direction again " 
		<< remainingZ ;
    }
    else {
      LogDebug("RKPropagatorInS")  << "Accuracy not reached yet, trying in opposite direction " 
		<< remainingZ ;
      currentDirection = invertDirection( currentDirection);
    }
    startZ = remainingZ;
  }

  edm::LogError("FailedPropagation") << " too many iterations trying to reach plane ";
  return GlobalParametersWithPath();
}


GlobalParametersWithPath
RKPropagatorInS::propagateParametersOnCylinder( const FreeTrajectoryState& ts, 
						const Cylinder& cyl) const {

  typedef RKAdaptiveSolver<double,RKOneCashKarpStep, 6>   Solver;
  typedef Solver::Vector                                  RKVector;
  
  
  GlobalPoint sp = cyl.position();
  if unlikely(sp.x()!=0. || sp.y()!=0.) {
      throw PropagationException("Cannot propagate to an arbitrary cylinder");
    }
  
  GlobalPoint gpos( ts.position());
  GlobalVector gmom( ts.momentum());
  LocalPoint pos(cyl.toLocal(gpos));
  LocalVector mom(cyl.toLocal(gmom));
  double startR = cyl.radius() - pos.perp();
  
  // LogDebug("RKPropagatorInS")  << "RKPropagatorInS: starting from FTS " << ts ;
  
  
  // (transverse) curvature
  double rho = ts.transverseCurvature();
  //
  // Straight line approximation? |rho|<1.e-10 equivalent to ~ 1um 
  // difference in transversal position at 10m.
  //
  if  unlikely( fabs(rho)<1.e-10 ) {
      //
      // Instantiate auxiliary object for finding intersection.
      // Frame-independant point and vector are created explicitely to 
      // avoid confusing gcc (refuses to compile with temporary objects
      // in the constructor).
      //
      
      StraightLineBarrelCylinderCrossing cylCrossing(gpos,gmom,propagationDirection());
      
      //
      // get solution
      //
      std::pair<bool,double> propResult = cylCrossing.pathLength(cyl);
      if  likely( propResult.first) {
	  double s = propResult.second;
	  // point (reconverted to GlobalPoint)
	  GlobalPoint x (cylCrossing.position(s));
	  GlobalTrajectoryParameters res( x, gmom, ts.charge(), &theVolume);
	  LogDebug("RKPropagatorInS")  << "Straight line propagation to cylinder succeeded !!";
	  return GlobalParametersWithPath( res, s);
	} 
      
      //do someting 
      edm::LogError("RKPropagatorInS")  <<"Straight line propagation to cylinder failed !!";
      return GlobalParametersWithPath( );
      
    }
  
  
  RKLocalFieldProvider field(theVolume, cyl);
  // StraightLineCylinderCrossing pathLength( pos, mom, propagationDirection());
  CartesianLorentzForce deriv(field, ts.charge());
  
  RKCartesianDistance dist;
  double eps = theTolerance;
  Solver solver;
  double stot = 0;
  PropagationDirection currentDirection = propagationDirection();
  
  RKVector start( CartesianStateAdaptor::rkstate( pos.basicVector(), mom.basicVector()));
  int safeGuard = 0;
  while (safeGuard++<100) {
    CartesianStateAdaptor startState(start);
    StraightLineCylinderCrossing pathLength( LocalPoint(startState.position()), 
					     LocalVector(startState.momentum()), 
					     currentDirection, eps);
    
    std::pair<bool,double> path = pathLength.pathLength( cyl);
    if  unlikely(!path.first) { 
	LogDebug("RKPropagatorInS")  << "RKPropagatorInS: Path length calculation to cylinder failed!" 
				     << "Radius " << cyl.radius() << " pos.perp() " << LocalPoint(startState.position()).perp() ;
	return GlobalParametersWithPath();
      }
    
    LogDebug("RKPropagatorInS")  << "RKPropagatorInS: Path lenght to cylinder is " << path.second 
				 << " from point (R,z) " << startState.position().perp() 
				 << ", " << startState.position().z()
				 << " to R " << cyl.radius() 
      ;
    
    
    double sstep = path.second;
    if  unlikely( std::abs(sstep) < eps) {
	LogDebug("RKPropagatorInS")  << "accuracy not reached, but pathLength calculation says we are there! "
				     << path.second ;
	
	GlobalTrajectoryParameters res( gtpFromLocal( startState.position(), 
						      startState.momentum(), 
						      ts.charge(), cyl));
	return GlobalParametersWithPath( res, stot);
      }
    
    LogDebug("RKPropagatorInS")  << "RKPropagatorInS: Solving for " << sstep 
				 << " current distance to cylinder is " << startR ;
    
    RKVector rkresult = solver( 0, start, sstep, deriv, dist, eps);
    stot += sstep;
    CartesianStateAdaptor cur( rkresult);
    double remainingR = cyl.radius() - cur.position().perp();

    if ( fabs(remainingR) < eps) {
      LogDebug("RKPropagatorInS")  << "Accuracy reached! " << remainingR ;
      GlobalTrajectoryParameters res( gtpFromLocal( cur.position(), 
						    cur.momentum(), 
						    ts.charge(), cyl));
      return GlobalParametersWithPath( res, stot);
    }
    
    start = rkresult;
    if (remainingR * startR > 0) {
      LogDebug("RKPropagatorInS")  << "Accuracy not reached yet, trying in same direction again " 
				   << remainingR ;
    }
    else {
      LogDebug("RKPropagatorInS")  << "Accuracy not reached yet, trying in opposite direction " 
				   << remainingR ;
      currentDirection = invertDirection( currentDirection);
    }
    startR = remainingR;
  }
  
  edm::LogError("FailedPropagation") << " too many iterations trying to reach cylinder ";
  return GlobalParametersWithPath();
}


Propagator * RKPropagatorInS::clone() const
{
    return new RKPropagatorInS(*this);
}

GlobalTrajectoryParameters RKPropagatorInS::gtpFromLocal( const Basic3DVector<double>& lpos,
							  const Basic3DVector<double>& lmom,
							  TrackCharge ch, const Surface& surf) const
{
    return GlobalTrajectoryParameters( surf.toGlobal( LocalPoint( lpos)),
				       surf.toGlobal( LocalVector( lmom)), ch, &theVolume);
}

GlobalTrajectoryParameters 
RKPropagatorInS::gtpFromVolumeLocal( const CartesianStateAdaptor& state, 
				     TrackCharge charge) const
{
  return GlobalTrajectoryParameters( globalPosition(state.position()), 
				     globalMomentum(state.momentum()), 
				     charge, &theVolume);
}
