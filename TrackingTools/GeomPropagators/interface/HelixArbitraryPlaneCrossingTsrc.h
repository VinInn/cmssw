#pragma once
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TwoFloat/include/TwoFloat.h>
#include <cmath>
#include <vdt/vdtMath.h>
#include <iostream>
#include "FWCore/Utilities/interface/Likely.h"

template<std::floating_point T>
inline void fast_sincosT( const T xx, T & s, T &c );

template<>
inline void fast_sincosT<float>( const float xx, float & s, float &c ) {
    vdt::fast_sincosf(xx,s,c);
}


template<>
inline void fast_sincosT<double>( const double xx, double & s, double &c ) {
    vdt::fast_sincos(xx,s,c);
}

template<typename T>
inline void fast_sincosT( const TwoFloat<T> xx, TwoFloat<T> & s, TwoFloat<T> &c ) {
   T sl,cl;
   fast_sincosT(T(xx),sl,cl);
   s= sl; c = cl;
}



#ifdef VI_DEBUG
#include <atomic>
struct MaxIter {
  MaxIter() {}
  ~MaxIter() { std::cout << "maxiter " << v << std::endl; }
  void operator()(int i) const {
    int old = v;
    int t = std::min(old, i);
    while (not v.compare_exchange_weak(old, t)) {
      t = std::min(old, i);
    }
  }
  mutable std::atomic<int> v{100};
};
#else
struct MaxIter {
  MaxIter() {}
  void operator()(int) const {}
};
#endif
static const MaxIter maxiter;


constexpr float theNumericalPrecision = 5.e-7f;
constexpr float theMaxDistToPlane = 1.e-4f;


template<typename T>
HelixArbitraryPlaneCrossingT<T>::HelixArbitraryPlaneCrossingT(const PositionType& point,
                                                         const DirectionType& direction,
                                                         const float curvature,
                                                         const PropagationDirection propDir)
    : theQuadraticCrossingFromStart(point, direction, curvature, propDir),
      theX0(point.x()),
      theY0(point.y()),
      theZ0(point.z()),
      theRho(curvature),
      thePropDir(propDir),
      theCachedS(0),
      theCachedDPhi(0.),
      theCachedSDPhi(0.),
      theCachedCDPhi(1.) {
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
  theSinTheta = pt2 * ptI * pI;
}
//
// Propagation status and path length to intersection
//
template<typename T>
std::pair<bool, double> HelixArbitraryPlaneCrossingT<T>::pathLength(const Plane& plane) {
  //
  // Constants used for control of convergence
  //
  constexpr int maxIterations = 20;
  //
  // maximum distance to plane (taking into account numerical precision)
  //
  float maxNumDz = theNumericalPrecision * plane.position().mag();
  float safeMaxDist = (theMaxDistToPlane > maxNumDz ? theMaxDistToPlane : maxNumDz);
  //
  // Prepare internal value of the propagation direction and position / direction vectors for iteration
  //

  float dz = plane.localZ(Surface::GlobalPoint(theX0, theY0, theZ0));
  if (std::abs(dz) < safeMaxDist)
    return std::make_pair(true, 0.);

  bool notFail;
  T dSTotal;
  // Use existing 2nd order object at first pass
  std::tie(notFail, dSTotal) = theQuadraticCrossingFromStart.pathLength(plane);
  if UNLIKELY (!notFail)
    return std::make_pair(notFail, dSTotal);
  auto xnew = positionInT(dSTotal);

  auto propDir = thePropDir;
  auto newDir = dSTotal >= 0 ? alongMomentum : oppositeToMomentum;
  if (propDir == anyDirection) {
    propDir = newDir;
  } else {
    if UNLIKELY (newDir != propDir)
      return std::pair<bool, T>(false, 0);
  }

  //
  // Prepare iterations: count and total pathlength
  //
  auto iteration = maxIterations;
  while (notAtSurface(plane, xnew, safeMaxDist)) {
    //
    // return empty solution vector if no convergence after maxIterations iterations
    //
    if UNLIKELY (--iteration == 0) {
      LogDebug("HelixArbitraryPlaneCrossingT") << "pathLength : no convergence";
      return std::pair<bool, T>(false, 0);
    }

    //
    // create temporary object for subsequent passes.
    auto pnew = directionInT(dSTotal);
    HelixArbitraryPlaneCrossingT2Order quadraticCrossing(
        xnew.x(), xnew.y(), xnew.z(), pnew.x(), pnew.y(), theCosTheta, theSinTheta, theRho, anyDirection);

    auto deltaS2 = quadraticCrossing.pathLength(plane);

    if UNLIKELY (!deltaS2.first)
      return deltaS2;
    //
    // Calculate and sort total pathlength (max. 2 solutions)
    //
    dSTotal += deltaS2.second;
    auto newDir = dSTotal >= 0 ? alongMomentum : oppositeToMomentum;
    if (propDir == anyDirection) {
      propDir = newDir;
    } else {
      if UNLIKELY (newDir != propDir)
        return std::pair<bool, T>(false, 0);
    }
    //
    // Step forward by dSTotal.
    //
    xnew = positionInT(dSTotal);
  }
  //
  // Return result
  //
  maxiter(iteration);

  return std::make_pair(true, dSTotal);
}
//
// Position on helix after a step of path length s.
//
template<typename T>
HelixPlaneCrossing::PositionType HelixArbitraryPlaneCrossingT<T>::position(double s) const {
  // use result in T precision
  PositionTypeT pos = positionInT(s);
  return PositionType(pos.x(), pos.y(), pos.z());
}
//
// Position on helix after a step of path length s in T precision.
//
template<typename T>
typename HelixArbitraryPlaneCrossingT<T>::PositionTypeT HelixArbitraryPlaneCrossingT<T>::positionInT(T s) const {
  constexpr T one = 1.;
  //
  // Calculate delta phi (if not already available)
  //
  if UNLIKELY (s != theCachedS) {
    theCachedS = s;
    theCachedDPhi = theCachedS * theRho * theSinTheta;
    fast_sincosT(theCachedDPhi, theCachedSDPhi, theCachedCDPhi);
  }
  //
  // Calculate with appropriate formulation of full helix formula or with
  //   2nd order approximation.
  //
  //    if ( fabs(theCachedDPhi)>1.e-1 ) {
  if (fabs(theCachedDPhi) > T(1.e-4)) {
    // "standard" helix formula
    T o = one / theRho;
    return PositionTypeT(theX0 + (-theSinPhi0 * (one - theCachedCDPhi) + theCosPhi0 * theCachedSDPhi) * o,
                              theY0 + (theCosPhi0 * (one - theCachedCDPhi) + theSinPhi0 * theCachedSDPhi) * o,
                              theZ0 + theCachedS * theCosTheta);
  }
  //    else if ( fabs(theCachedDPhi)>theNumericalPrecision ) {
  //      // full helix formula, but avoiding (1-cos(deltaPhi)) for small angles
  //      return PositionTypeT(theX0+(-theSinPhi0*theCachedSDPhi*theCachedSDPhi/(1.+theCachedCDPhi)+
  //  				     theCosPhi0*theCachedSDPhi)/theRho,
  //  			      theY0+(theCosPhi0*theCachedSDPhi*theCachedSDPhi/(1.+theCachedCDPhi)+
  //  				     theSinPhi0*theCachedSDPhi)/theRho,
  //  			      theZ0+theCachedS*theCosTheta);
  //    }
  else {
    // Use 2nd order.
    return theQuadraticCrossingFromStart.positionInT(theCachedS);
  }
}
//
// Direction vector on helix after a step of path length s.
//
template<typename T>
HelixPlaneCrossing::DirectionType HelixArbitraryPlaneCrossingT<T>::direction(double s) const {
  // use result in T precision
  DirectionTypeT dir = directionInT(s);
  return DirectionType(dir.x(), dir.y(), dir.z());
}
//
// Direction vector on helix after a step of path length s in T precision.
//
template<typename T>
typename HelixArbitraryPlaneCrossingT<T>::DirectionTypeT HelixArbitraryPlaneCrossingT<T>::directionInT(T s) const {
  //
  // Calculate delta phi (if not already available)
  //
  if UNLIKELY (s != theCachedS) {  // very very unlikely!
    theCachedS = s;
    theCachedDPhi = theCachedS * theRho * theSinTheta;
    fast_sincosT(theCachedDPhi, theCachedSDPhi, theCachedCDPhi);
  }

  if (fabs(theCachedDPhi) > T(1.e-4)) {
    // full helix formula
    return DirectionTypeT(theCosPhi0 * theCachedCDPhi - theSinPhi0 * theCachedSDPhi,
                               theSinPhi0 * theCachedCDPhi + theCosPhi0 * theCachedSDPhi,
                               theCosTheta / theSinTheta);
  } else {
    // 2nd order
    return theQuadraticCrossingFromStart.directionInT(theCachedS);
  }
}
//   Iteration control: continue if distance to plane > theMaxDistToPlane. Includes
//   protection for numerical precision (Surfaces work with single precision).
template<typename T>
bool HelixArbitraryPlaneCrossingT<T>::notAtSurface(const Plane& plane,
                                               const PositionTypeT& point,
                                               const float maxDist) const {
  float dz = plane.localZ(Surface::GlobalPoint(point.x(), point.y(), point.z()));
  return std::abs(dz) > maxDist;
}

