#ifndef PathToPlane2Order_H
#define PathToPlane2Order_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"


/** Computes the path length to reach a plane in general magnetic field.
 *  The problem (starting state and plane) is transformed to a frame where the
 *  starting field is along Z, and the AnalyticalHelixPlaneCrossing is used
 *  to compute the path length.
 */


class dso_internal PathToPlane2Order {
public:

    typedef Plane::Scalar                                Scalar;
    typedef Basic3DVector<Scalar>                        Vector3D;
    typedef GloballyPositioned<Scalar>                   Frame;

    explicit PathToPlane2Order( const MagVolume & fld) : 
      theField(fld) {}

    /// the position and momentum are local in the FieldFrame;
    /// the plane is in the global frame
    std::pair<bool,double> operator()( const HessianPlane<double>& plane, 
				       const Vector3D& position,
				       const Vector3D& momentum,
				       float charge,
				       const PropagationDirection propDir = alongMomentum) const;

  Vector3D inTesla( Vector3D lp) const {
    return theField.fieldInTesla( LocalPoint(lp) ).basicVector();
  }

private:
    const MagVolume & theField;
};

#endif
