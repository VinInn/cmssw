#ifndef HELIXARBITRARYPLANECROSSING2ORDER_H_
#define HELIXARBITRARYPLANECROSSING2ORDER_H_
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/HelixPlaneCrossing.h"
#include "FWCore/Utilities/interface/Visibility.h"

/** Calculates intersections of a helix with planes of
 *  any orientation using a parabolic approximation. */

class HelixArbitraryPlaneCrossing2Order GCC11_FINAL : public HelixPlaneCrossing {
public:
 // double precision vectors
  //
  typedef Basic3DVector<double>  PositionTypeDouble;
  typedef Basic3DVector<double>  DirectionTypeDouble;

  /** Constructor using point, direction and (transverse!) curvature.
   */
  HelixArbitraryPlaneCrossing2Order(const PositionType& point,
				    const DirectionType& direction,
				    const float curvature,
				    const PropagationDirection propDir = alongMomentum);

  /** Fast constructor (for use by HelixArbitraryPlaneCrossing).
   */
  HelixArbitraryPlaneCrossing2Order(PositionTypeDouble const & pos,
				    double cosPhi0, double sinPhi0,
				    double cosTheta, double sinTheta,
				    double rho,
				    const PropagationDirection propDir = alongMomentum) :
    thePos(pos),
    theCosPhi0(cosPhi0), theSinPhi0(sinPhi0),
    theCosTheta(cosTheta), theSinThetaI(1./sinTheta),
    theRho(rho), 
    thePropDir(propDir) {}

  // destructor
  virtual ~HelixArbitraryPlaneCrossing2Order() {}

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the plane. The 
   *  starting point is given in the constructor.
   */
  virtual std::pair<bool,double> pathLength(const Plane&);

  /** Position at pathlength s from the starting point.
   */
  virtual PositionType position(double s) const;

  /** Direction at pathlength s from the starting point.
   */
  virtual DirectionType direction(double s) const;

  /** Position at pathlength s from the starting point in double precision.
   */
  PositionTypeDouble positionInDouble(double s) const;

  /** Direction at pathlength s from the starting point in double precision.
   */
  DirectionTypeDouble directionInDouble(double s) const;

  /** Pathlength to closest solution.
   */
  inline double smallestPathLength (const double firstPathLength,
				    const double secondPathLength) const {
    return fabs(firstPathLength)<fabs(secondPathLength) ? firstPathLength : secondPathLength;
  }

private:

  /** Choice of one of two solutions according to the propagation direction.
   */
  std::pair<bool,double> solutionByDirection(const double dS1,const double dS2) const dso_internal;

private:
  const PositionTypeDouble thePos;
  double theCosPhi0,theSinPhi0;
  double theCosTheta,theSinThetaI;
  const double theRho;
  const PropagationDirection thePropDir;

};

#endif


