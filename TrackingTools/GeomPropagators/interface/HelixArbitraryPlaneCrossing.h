#ifndef HELIXARBITRARYPLANECROSSING_H_
#define HELIXARBITRARYPLANECROSSING_H_
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/HelixPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing2Order.h"
#include "FWCore/Utilities/interface/Visibility.h"

/** Calculates intersections of a helix with planes of any orientation. */

class HelixArbitraryPlaneCrossing GCC11_FINAL : public HelixPlaneCrossing {
public:
  /** Constructor using point, direction and (transverse!) curvature.
   */
  HelixArbitraryPlaneCrossing(const PositionType& point,
				    const DirectionType& direction,
				    const float curvature,
			            const PropagationDirection propDir = alongMomentum);
  // destructor
  virtual ~HelixArbitraryPlaneCrossing() {}

  //
  // double precision vectors for internal use
  //
  typedef Basic3DVector<double>  PositionTypeDouble;
  typedef Basic3DVector<double>  DirectionTypeDouble;


  PositionTypeDouble position0() const { return theQuadraticCrossingFromStart.position0();}
  DirectionTypeDouble direction0() const { return theQuadraticCrossingFromStart.direction0();}
  double sinThetaI() const { return theQuadraticCrossingFromStart.sinThetaI(); }
  double rho() const { return theQuadraticCrossingFromStart.rho();}
  PropagationDirection propDir0() const { return theQuadraticCrossingFromStart.propDir0();}



  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the plane. The 
   *  starting point is given in the constructor.
   */
  std::pair<bool,double> pathLength( const Plane& p) override { return pathLength(p.hessianPlaneDouble());}
  std::pair<bool,double> pathLength(HPlane const&) override;

  /** Position at pathlength s from the starting point.
   */
  virtual PositionType position(double s) const;

  /** Direction at pathlength s from the starting point.
   */
  virtual DirectionType direction(double s) const;


private:

  /** Position at pathlength s from the starting point.
   */
  PositionTypeDouble positionInDouble(double s) const dso_internal;

  /** Direction at pathlength s from the starting point.
   */
  DirectionTypeDouble directionInDouble(double s) const dso_internal;

private:
  HelixArbitraryPlaneCrossing2Order theQuadraticCrossingFromStart;

  double theCosTheta,theSinTheta;

  mutable double theCachedS;
  mutable double theCachedDPhi;
  mutable double theCachedSDPhi;
  mutable double theCachedCDPhi;

};

#endif
