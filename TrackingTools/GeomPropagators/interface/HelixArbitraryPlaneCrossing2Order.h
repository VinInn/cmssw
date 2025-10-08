#ifndef HELIXARBITRARYPLANECROSSING2ORDER_H_
#define HELIXARBITRARYPLANECROSSING2ORDER_H_
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/HelixPlaneCrossing.h"
#include "FWCore/Utilities/interface/Visibility.h"

/** Calculates intersections of a helix with planes of
 *  any orientation using a parabolic approximation. */

template<typename T>
class HelixArbitraryPlaneCrossingT2Order final : public HelixPlaneCrossing {
public:
  /** Constructor using point, direction and (transverse!) curvature.
   */
  HelixArbitraryPlaneCrossingT2Order(const PositionType& point,
                                    const DirectionType& direction,
                                    const float curvature,
                                    const PropagationDirection propDir = alongMomentum);
  /** Fast constructor (for use by HelixArbitraryPlaneCrossing).
   */
  HelixArbitraryPlaneCrossingT2Order(const T& x0,
                                    const T& y0,
                                    const T& z0,
                                    const T& cosPhi0,
                                    const T& sinPhi0,
                                    const T& cosTheta,
                                    const T& sinTheta,
                                    const T& rho,
                                    const PropagationDirection propDir = alongMomentum)
      : theX0(x0),
        theY0(y0),
        theZ0(z0),
        theCosPhi0(cosPhi0),
        theSinPhi0(sinPhi0),
        theCosTheta(cosTheta),
        theSinThetaI(1. / sinTheta),
        theRho(rho),
        thePropDir(propDir) {}

  // destructor
  ~HelixArbitraryPlaneCrossingT2Order() override {}

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the plane. The 
   *  starting point is given in the constructor.
   */
  std::pair<bool, double> pathLength(const Plane&) override;

  /** Position at pathlength s from the starting point.
   */
  PositionType position(double s) const override;

  /** Direction at pathlength s from the starting point.
   */
  DirectionType direction(double s) const override;
  //
  // T precision vectors
  //
  typedef Basic3DVector<T> PositionTypeT;
  typedef Basic3DVector<T> DirectionTypeT;

  /** Position at pathlength s from the starting point in T precision.
   */
  PositionTypeT positionInT(T s) const;

  /** Direction at pathlength s from the starting point in T precision.
   */
  DirectionTypeT directionInT(T s) const;

  /** Pathlength to closest solution.
   */
  inline T smallestPathLength(const T firstPathLength, const T secondPathLength) const {
    return fabs(firstPathLength) < fabs(secondPathLength) ? firstPathLength : secondPathLength;
  }

private:
  /** Choice of one of two solutions according to the propagation direction.
   */
  std::pair<bool, T> solutionByDirection(const T dS1, const T dS2) const dso_internal;

private:
  const T theX0, theY0, theZ0;
  T theCosPhi0, theSinPhi0;
  T theCosTheta, theSinThetaI;
  const T theRho;
  const PropagationDirection thePropDir;
};

#include "HelixArbitraryPlaneCrossingT2OrderSrc.h"

using HelixArbitraryPlaneCrossing2Order = HelixArbitraryPlaneCrossingT2Order<double>;

#endif
