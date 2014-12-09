#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing2Order.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/SIMDVec.h"

#include <cmath>
#include <cfloat>
#include "FWCore/Utilities/interface/Likely.h"

HelixArbitraryPlaneCrossing2Order::HelixArbitraryPlaneCrossing2Order(const PositionType& point,
								     const DirectionType& direction,
								     const float curvature,
								     const PropagationDirection propDir) :
  thePos(point),
  theDir0(direction),
  theRho(curvature),
  thePropDir(propDir)
{
  //
  // Components of direction vector (with correct normalisation)
  //
  auto pt2 = theDir0.perp2();
  auto pz = theDir0.z();
  double p2 = pt2+pz*pz;
  double pI = 1./sqrt(p2);
  double ptI = 1./sqrt(pt2);
  theDir0 *=ptI;
  theSinThetaI = p2*pI*ptI; //  (1/(pt/p)) = p/pt = p*ptI and p = p2/p = p2*pI
}

//
// Propagation status and path length to intersection
//
std::pair<bool,double>
HelixArbitraryPlaneCrossing2Order::pathLength(HPlane const& plane) {
  //
  // get local z-vector in global co-ordinates and
  // distance to starting point
  //
  auto && nv = plane.basicVector(); 
  double cP = plane.localZ(thePos);
  //
  // coefficients of 2nd order equation to obtain intersection point
  // in approximation (without curvature-related factors).
  //
  double ceq1 = theRho*(nv.x()*theDir0.y() - nv.y()*theDir0.x());
  double ceq2 = nv.dot(theDir0);
  double ceq3 = cP;
  //
  // Check for degeneration to linear equation (zero
  //   curvature, forward plane or direction perp. to plane)
  //
  Vector2D dS;
  if likely( std::abs(ceq1)>FLT_MIN ) {
    double deq1 = ceq2*ceq2;
    double deq2 = ceq1*ceq3;
    //
    // Standard solution for quadratic equations
    //
    auto deq = deq1+2*deq2;
    if unlikely( deq<0. )  return std::pair<bool,double>(false,0);
    auto ceq =  ceq2+std::copysign(std::sqrt(deq),ceq2);
    // dS[0] =     (ceq/ceq1)*theSinThetaI;
    // dS[1] = -2.*(ceq3/ceq)*theSinThetaI;
    Vector2D c1 = {ceq,-2*ceq3};
    Vector2D c2 = {ceq1,ceq};
    dS = (c1/c2)*theSinThetaI;
  }
  else {
    //
    // Special case: linear equation
    //
    dS[0] = dS[1] = -(ceq3/ceq2)*theSinThetaI;
  }
  //
  // Choose and return solution
  //
  return solutionByDirection(dS[0],dS[1]);

}

//
// Position after a step of path length s (2nd order)
//
HelixPlaneCrossing::PositionType
HelixArbitraryPlaneCrossing2Order::position (double s) const {
  // use double precision result
  return positionInDouble(s);
}

//
// Position after a step of path length s (2nd order) (in double precision)
//
HelixArbitraryPlaneCrossing2Order::PositionTypeDouble
HelixArbitraryPlaneCrossing2Order::positionInDouble (double s) const {
  // based on path length in the transverse plane
  double st = s/theSinThetaI;
  return thePos + st*(theDir0 + (st*0.5*theRho)*PositionTypeDouble(-theDir0.y(),+theDir0.x(),0) );
}

//
// Direction after a step of path length 2 (2nd order) (in double precision)
//
HelixPlaneCrossing::DirectionType
HelixArbitraryPlaneCrossing2Order::direction (double s) const {
  // use double precision result
  return directionInDouble(s);
}

//
// Direction after a step of path length 2 (2nd order)
//
HelixArbitraryPlaneCrossing2Order::DirectionTypeDouble
HelixArbitraryPlaneCrossing2Order::directionInDouble (double s) const {
  // based on delta phi
  double dph = s*theRho/theSinThetaI;
  return theDir0 +dph*DirectionTypeDouble(-(theDir0.y()+0.5*dph*theDir0.x()),
			                  +(theDir0.x()-0.5*dph*theDir0.y()),
			                  0
                                         );
}

//
// Choice of solution according to propagation direction
//
std::pair<bool,double>
HelixArbitraryPlaneCrossing2Order::genericSolutionByDirection(const double dS1,
						       const double dS2) const {
  bool valid = false;
  double path = 0;
    // use same logic for both directions (invert if necessary)
    double propSign = thePropDir==alongMomentum ? 1 : -1;
    double s1(propSign*dS1);
    double s2(propSign*dS2);
    // sort
    if ( s1 > s2 ) std::swap(s1,s2);
    // choose solution (if any with positive sign)
    if ( (s1<0) & (s2>=0) ) {
      // First solution in backward direction: choose second one.
      valid = true;
      path = propSign*s2;
    }
    else if ( s1>=0 ) {
      // First solution in forward direction: choose it (s2 is further away!).
      valid = true;
      path = propSign*s1;
    }
  return std::pair<bool,double>(valid,path);
}
