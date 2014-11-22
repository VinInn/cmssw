#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing2Order.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <cmath>
#include <cfloat>
#include "FWCore/Utilities/interface/Likely.h"

HelixArbitraryPlaneCrossing2Order::HelixArbitraryPlaneCrossing2Order(const PositionType& point,
								     const DirectionType& direction,
								     const float curvature,
								     const PropagationDirection propDir) :
  thePos(point),
  theRho(curvature),
  thePropDir(propDir)
{
  //
  // Components of direction vector (with correct normalisation)
  //
  double px = direction.x();
  double py = direction.y();
  double pz = direction.z();
  double pt2 = px*px+py*py;
  double p2 = pt2+pz*pz;
  double pI = 1./sqrt(p2);
  double ptI = 1./sqrt(pt2);
  theCosPhi0 = px*ptI;
  theSinPhi0 = py*ptI;
  theCosTheta = pz*pI;
  theSinThetaI = p2*pI*ptI; //  (1/(pt/p)) = p/pt = p*ptI and p = p2/p = p2*pI
}

//
// Propagation status and path length to intersection
//
std::pair<bool,double>
HelixArbitraryPlaneCrossing2Order::pathLength(const Plane& plane) {
  //
  // get local z-vector in global co-ordinates and
  // distance to starting point
  //
  auto const & normalToPlane = plane.normalVector();
  double nPx = normalToPlane.x();
  double nPy = normalToPlane.y();
  double nPz = normalToPlane.z();
  double cP = plane.localZ(GlobalPoint(thePos));
  //
  // coefficients of 2nd order equation to obtain intersection point
  // in approximation (without curvature-related factors).
  //
  double ceq1 = theRho*(nPx*theSinPhi0-nPy*theCosPhi0);
  double ceq2 = nPx*theCosPhi0 + nPy*theSinPhi0 + nPz*theCosTheta*theSinThetaI;
  double ceq3 = cP;
  //
  // Check for degeneration to linear equation (zero
  //   curvature, forward plane or direction perp. to plane)
  //
  double dS[2];
  if likely( std::abs(ceq1)>FLT_MIN ) {
    double deq1 = ceq2*ceq2;
    double deq2 = ceq1*ceq3;
    //
    // Standard solution for quadratic equations
    //
    auto deq = deq1+2*deq2;
    if unlikely( deq<0. )  return std::pair<bool,double>(false,0);
    auto ceq =  ceq2+std::copysign(std::sqrt(deq),ceq2);
    dS[0] =     (ceq/ceq1)*theSinThetaI;
    dS[1] = -2.*(ceq3/ceq)*theSinThetaI;
    /* does not vectorize anyhow...
    double c1[2] = {ceq,-2*ceq3};
    double c2[2] = {ceq1,ceq};
    for(auto i=0; i<2;++i) dS[i]=(c1[i]/c2[i])*theSinThetaI;
    */
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
  return thePos + PositionTypeDouble((theCosPhi0-(st*0.5*theRho)*theSinPhi0)*st,
			             (theSinPhi0+(st*0.5*theRho)*theCosPhi0)*st,
			             st*theCosTheta*theSinThetaI
                                    );
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
  return DirectionTypeDouble(theCosPhi0-(theSinPhi0+0.5*dph*theCosPhi0)*dph,
			     theSinPhi0+(theCosPhi0-0.5*dph*theSinPhi0)*dph,
			     theCosTheta*theSinThetaI);
}

//
// Choice of solution according to propagation direction
//
std::pair<bool,double>
HelixArbitraryPlaneCrossing2Order::solutionByDirection(const double dS1,
						       const double dS2) const {
  bool valid = false;
  double path = 0;
  if ( thePropDir == anyDirection ) {
    valid = true;
    path = smallestPathLength(dS1,dS2);
  }
  else {
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
  }
  return std::pair<bool,double>(valid,path);
}
