#ifndef HELIXARBITRARYPLANECROSSING_H_
#define HELIXARBITRARYPLANECROSSING_H_
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/GeomPropagators/interface/HelixPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing2Order.h"
#include "FWCore/Utilities/interface/Visibility.h"

/** Calculates intersections of a helix with planes of any orientation. */

template<typename T>
class HelixArbitraryPlaneCrossingT final : public HelixPlaneCrossing {
public:
  /** Constructor using point, direction and (transverse!) curvature.
   */
  HelixArbitraryPlaneCrossingT(const PositionType& point,
                              const DirectionType& direction,
                              const float curvature,
                              const PropagationDirection propDir = alongMomentum);
  // destructor
  ~HelixArbitraryPlaneCrossingT() override {}

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the plane. The 
   *  starting point is given in the constructor.
   */
  std::pair<bool, double> pathLength(const Plane& plane) override;

  /** Position at pathlength s from the starting point.
   */
  PositionType position(double s) const override;

  /** Direction at pathlength s from the starting point.
   */
  DirectionType direction(double s) const override;
  //
  // T precision vectors for internal use
  //
  typedef Basic3DVector<T> PositionTypeT;
  typedef Basic3DVector<T> DirectionTypeT;

  /** Position at pathlength s from the starting point.
   */
  PositionTypeT positionInT(T s) const;

  /** Direction at pathlength s from the starting point.
   */
  DirectionTypeT directionInT(T s) const;

private:
  /** Iteration control: check for significant distance to plane.
   */
  inline bool notAtSurface(const Plane&, const PositionTypeT&, const float) const dso_internal;

private:
  HelixArbitraryPlaneCrossingT2Order<T> theQuadraticCrossingFromStart;

  const T theX0, theY0, theZ0;
  T theCosPhi0, theSinPhi0;
  T theCosTheta, theSinTheta;
  const T theRho;

  const PropagationDirection thePropDir;

  mutable T theCachedS;
  mutable T theCachedDPhi;
  mutable T theCachedSDPhi;
  mutable T theCachedCDPhi;

};

#include  "HelixArbitraryPlaneCrossingTsrc.h"

using HelixArbitraryPlaneCrossing = HelixArbitraryPlaneCrossingT<double>;

#endif
