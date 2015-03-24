#ifndef RKframeChanger_H
#define RKframeChanger_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

namespace frameChanger dso_internal {

  /** Moves the first argument ("plane") to the reference frame given by the second
   *  argument ("frame"). The returned frame is not positioned globally!
   */
    template <typename T, typename U>
    static
    HessianPlane<T> transform( const HessianPlane<T>& plane, const GloballyPositioned<U>& frame) {
        using Frame = GloballyPositioned<U>;
        using GlobalVector = typename GloballyPositioned<U>::GlobalVector; 
        return HessianPlane<T>(frame.toLocal(GlobalVector(plane.basicVector())).basicVector(), 
                               plane.localZ(frame.position().basicVector()) 
                              );
    }

  /** Moves the first argument ("plane") to the reference frame given by the second 
   *  argument ("frame"). The returned frame is not positioned globally!
   */
    template <typename T>
    static
    Plane transformPlane( const Plane& plane, const GloballyPositioned<T>& frame) {
        typedef GloballyPositioned<T>                  Frame;
	typename Plane::RotationType rot = plane.rotation() * frame.rotation().transposed();
	typename Frame::LocalPoint lpos = frame.toLocal(plane.position());
	typename Plane::PositionType pos( lpos.basicVector()); // cheat!
	return Plane(pos, rot);
    }


/** Moves the first argument ("plane") to the reference frame given by the second 
 *  argument ("frame"). The returned frame is not positioned globally!
 */
    template <typename T, typename U>
    static
    GloballyPositioned<T> toFrame( const GloballyPositioned<T>& plane, 
				   const GloballyPositioned<U>& frame) {
	typedef GloballyPositioned<T>                  Plane;
	typedef GloballyPositioned<U>                  Frame;

	typename Plane::RotationType rot = plane.rotation() * frame.rotation().transposed();
	typename Frame::LocalPoint lpos = frame.toLocal(plane.position());
	typename Plane::PositionType pos( lpos.basicVector()); // cheat!
	return Plane( pos, rot);

    }

};

#endif
