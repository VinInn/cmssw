#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"


#include <cmath>
#include <cfloat>

StraightLinePlaneCrossing::StraightLinePlaneCrossing(const PositionType& point,
						     const DirectionType& momentum,
						     const PropagationDirection propDir) :
  theX0(point),
  theP0(momentum.unit()),
  thePropDir(propDir) {
//   cout << "StraightLinePlaneCrossing: x0 = " << point
//        << ", p0 = " << momentum << endl;
}

//
// Propagation status  and path length to intersection
//
std::pair<bool,double>
StraightLinePlaneCrossing::pathLength (const HPlane& plane) const {
  //
  // Protect against p_normal=0 and calculate path length
  //
  auto planeNormal = plane.basicVector();

  double pz = planeNormal.dot(theP0);
//   cout << "pz = " << pz << endl;
  if ( fabs(pz)<FLT_MIN )  return std::pair<bool,double>(false,0);

  double dS = -plane.localZ(theX0)/pz;

  if ( (thePropDir==alongMomentum && dS<0.) ||
       (thePropDir==oppositeToMomentum && dS>0.) )  return std::pair<bool,double>(false,0);
  //
  // Return result
  //
  return std::pair<bool,double>(true,dS);
}

std::pair<bool,StraightLinePlaneCrossing::PositionType> 
StraightLinePlaneCrossing::position(const Plane& plane) const
{
    std::pair<bool,double> crossed = pathLength(plane);
    if (crossed.first) return std::pair<bool,PositionType>(true, position(crossed.second));
    else return std::pair<bool,PositionType>(false, PositionType());
}

