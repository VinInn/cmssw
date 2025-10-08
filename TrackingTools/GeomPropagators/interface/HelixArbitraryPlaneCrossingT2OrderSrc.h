#pragma once
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <TwoFloat/include/TwoFloat.h>
#include <cmath>
#include <cfloat>
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/isFinite.h"

template<typename T>
HelixArbitraryPlaneCrossingT2Order<T>::HelixArbitraryPlaneCrossingT2Order(const PositionType& point,
                                                                     const DirectionType& direction,
                                                                     const float curvature,
                                                                     const PropagationDirection propDir)
    : theX0(point.x()), theY0(point.y()), theZ0(point.z()), theRho(curvature), thePropDir(propDir) {
  //
  // Components of direction vector (with correct normalisation)
  //
  T px = direction.x();
  T py = direction.y();
  T pz = direction.z();
  T pt2 = px * px + py * py;
  T p2 = pt2 + pz * pz;
  T pI = rsqrt(p2);
  T ptI = rsqrt(pt2);
  theCosPhi0 = px * ptI;
  theSinPhi0 = py * ptI;
  theCosTheta = pz * pI;
  theSinThetaI = p2 * pI * ptI;  //  (1/(pt/p)) = p/pt = p*ptI and p = p2/p = p2*pI
}

//
// Propagation status and path length to intersection
//
template<typename T>
std::pair<bool, double> HelixArbitraryPlaneCrossingT2Order<T>::pathLength(const Plane& plane) {
  //
  // get local z-vector in global co-ordinates and
  // distance to starting point
  //
  GlobalVector normalToPlane = plane.normalVector();
  T nPx = normalToPlane.x();
  T nPy = normalToPlane.y();
  T nPz = normalToPlane.z();
  T cP = plane.localZ(GlobalPoint(theX0, theY0, theZ0));
  //
  // coefficients of 2nd order equation to obtain intersection point
  // in approximation (without curvature-related factors).
  //
  T ceq1 = theRho * (nPx * theSinPhi0 - nPy * theCosPhi0);
  T ceq2 = nPx * theCosPhi0 + nPy * theSinPhi0 + nPz * theCosTheta * theSinThetaI;
  T ceq3 = cP;
  //
  // Check for degeneration to linear equation (zero
  //   curvature, forward plane or direction perp. to plane)
  //
  T dS1, dS2;
  if LIKELY (fabs(ceq1) > FLT_MIN) {
    T deq1 = ceq2 * ceq2;
    T deq2 = ceq1 * ceq3;
    if (fabs(deq1) < FLT_MIN || fabs(deq2 / deq1) > T(1.e-6)) {
      //
      // Standard solution for quadratic equations
      //
      T deq = deq1 + T(2) * deq2;
      if UNLIKELY (deq < 0.)
        return std::pair<bool, double>(false, 0);
      T ceq = ceq2 + std::copysign(std::sqrt(deq), ceq2);
      dS1 = (ceq / ceq1) * theSinThetaI;
      dS2 = -T(2) * (ceq3 / ceq) * theSinThetaI;
    } else {
      //
      // Solution by expansion of sqrt(1+deq)
      //
      T ceq = (ceq2 / ceq1) * theSinThetaI;
      T deq = deq2 / deq1;
      deq *= (T(1) - T(0.5) * deq);
      dS1 = -ceq * deq;
      dS2 = ceq * (T(2) + deq);
    }
  } else {
    //
    // Special case: linear equation
    //
    dS1 = dS2 = -(ceq3 / ceq2) * theSinThetaI;
  }
  //
  // Choose and return solution
  //
  return solutionByDirection(dS1, dS2);
}

//
// Position after a step of path length s (2nd order)
//
template<typename T>
HelixPlaneCrossing::PositionType HelixArbitraryPlaneCrossingT2Order<T>::position(double s) const {
  // use T precision result
  PositionTypeT pos = positionInT(s);
  return PositionType(pos.x(), pos.y(), pos.z());
}
//
// Position after a step of path length s (2nd order) (in T precision)
//
template<typename T>
typename HelixArbitraryPlaneCrossingT2Order<T>::PositionTypeT 
HelixArbitraryPlaneCrossingT2Order<T>::positionInT(T s) const {
  // based on path length in the transverse plane
  T st = s / theSinThetaI;
  return PositionTypeT(theX0 + (theCosPhi0 - (st * T(0.5) * theRho) * theSinPhi0) * st,
                            theY0 + (theSinPhi0 + (st * T(0.5) * theRho) * theCosPhi0) * st,
                            theZ0 + st * theCosTheta * theSinThetaI);
}
//
// Direction after a step of path length 2 (2nd order) (in T precision)
//
template<typename T>
HelixPlaneCrossing::DirectionType HelixArbitraryPlaneCrossingT2Order<T>::direction(double s) const {
  // use T precision result
  DirectionTypeT dir = directionInT(s);
  return DirectionType(dir.x(), dir.y(), dir.z());
}
//
// Direction after a step of path length 2 (2nd order)
//
template<typename T>
typename HelixArbitraryPlaneCrossingT2Order<T>::DirectionTypeT 
HelixArbitraryPlaneCrossingT2Order<T>::directionInT(T s) const {
  // based on delta phi
  T dph = s * theRho / theSinThetaI;
  return DirectionTypeT(theCosPhi0 - (theSinPhi0 + T(0.5) * dph * theCosPhi0) * dph,
                             theSinPhi0 + (theCosPhi0 - T(0.5) * dph * theSinPhi0) * dph,
                             theCosTheta * theSinThetaI);
}
//
// Choice of solution according to propagation direction
//
template<typename T>
std::pair<bool, T> HelixArbitraryPlaneCrossingT2Order<T>::solutionByDirection(const T dS1,
                                                                               const T dS2) const {
  bool valid = false;
  T path = 0;
  if (thePropDir == anyDirection) {
    valid = true;
    path = smallestPathLength(dS1, dS2);
  } else {
    // use same logic for both directions (invert if necessary)
    T propSign = thePropDir == alongMomentum ? 1 : -1;
    T s1(propSign * dS1);
    T s2(propSign * dS2);
    // sort
    if (s1 > s2)
      std::swap(s1, s2);
    // choose solution (if any with positive sign)
    if ((s1 < 0) & (s2 >= 0)) {
      // First solution in backward direction: choose second one.
      valid = true;
      path = propSign * s2;
    } else if (s1 >= 0) {
      // First solution in forward direction: choose it (s2 is further away!).
      valid = true;
      path = propSign * s1;
    }
  }
  if (edm::isNotFinite(toSingle(path)))
    valid = false;
  return std::pair<bool, T>(valid, path);
}
