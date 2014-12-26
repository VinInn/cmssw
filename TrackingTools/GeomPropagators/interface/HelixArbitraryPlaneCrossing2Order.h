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
  typedef Basic2DVector<double>::MathVector  Vector2D;

  /** Constructor using point, direction and (transverse!) curvature.
   */
  HelixArbitraryPlaneCrossing2Order(const PositionType& point,
				    const DirectionType& direction,
				    const float curvature,
				    const PropagationDirection propDir = alongMomentum);

  /** Fast constructor (for use by HelixArbitraryPlaneCrossing).
   */
  HelixArbitraryPlaneCrossing2Order(PositionTypeDouble const & pos,
				    DirectionTypeDouble const & dir,
				    double sinThetaI,
				    double rho) :
    thePos(pos),
    theDir0(dir), theSinThetaI(sinThetaI),
    theRho(rho), 
    thePropDir(anyDirection) {}

  // destructor
  virtual ~HelixArbitraryPlaneCrossing2Order() {}

  PositionTypeDouble position0() const { return thePos;}
  DirectionTypeDouble direction0() const { return theDir0;}
  double sinThetaI() const { return theSinThetaI; }
  double rho() const { return theRho;} 
  PropagationDirection propDir0() const { return thePropDir;}




  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the plane. The 
   *  starting point is given in the constructor.
   */
  std::pair<bool,double> pathLength(const Plane& p) override { return pathLength(p.hessianPlaneDouble());}
  std::pair<bool,double> pathLength(HPlane const&)  override;

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
    return std::abs(firstPathLength)<std::abs(secondPathLength) ? firstPathLength : secondPathLength;
  }

private:

  /** Choice of one of two solutions according to the propagation direction.
   */
  inline
  std::pair<bool,double> solutionByDirection(const double dS1,const double dS2) const {
    if likely( thePropDir == anyDirection ) return std::make_pair(true,smallestPathLength(dS1,dS2));
    return genericSolutionByDirection(dS1,dS2);
  }

  std::pair<bool,double> genericSolutionByDirection(const double dS1,const double dS2) const dso_internal;



private:
  const PositionTypeDouble thePos;
  DirectionTypeDouble theDir0;
  double theSinThetaI;
  const double theRho;
  const PropagationDirection thePropDir;

};

#endif


