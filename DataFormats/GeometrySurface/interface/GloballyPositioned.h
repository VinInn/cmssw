#ifndef Geom_GloballyPositioned_H
#define Geom_GloballyPositioned_H

#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"
#include "DataFormats/GeometryVector/interface/LocalTag.h"
#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"

/** Base class for surfaces and volumes positioned in global 3D space.
 *  This class defines a cartesian reference frame, called in the 
 *  following the local frame.
 *  It provides position, orientation, and frame transformations for
 *  points and vectors.
 */

template <class T>
class GloballyPositioned {
public:

  typedef T                             Scalar;
  typedef Point3DBase<T,GlobalTag>      PositionType;
  typedef TkRotation<T>                 RotationType;
  typedef Point3DBase<T,GlobalTag>      GlobalPoint;
  typedef Point3DBase<T,LocalTag>       LocalPoint;
  typedef Vector3DBase<T,GlobalTag>     GlobalVector;
  typedef Vector3DBase<T,LocalTag>      LocalVector;

  static T iniPhi() {
    return 999.9978;
  }
  static T iniEta() {
    return 999.9978;
  }

  GloballyPositioned() {setCache();}
  GloballyPositioned( const PositionType& pos, const RotationType& rot) :
    thePos(pos), theRot(rot) {setCache();}

  virtual ~GloballyPositioned() {}

  const PositionType& position() const { return thePos;}

  const RotationType& rotation() const { return theRot;}

  T phi() const {
    return thePhi;
  }
  T eta() const { 
    return theEta;
  }
  // normal distance of origin
  T dv() const {
    return theDV;
  }

  // multiply inverse is faster
  class ToLocal {
  public:
    ToLocal(GloballyPositioned const & frame) :
      thePos(frame.position()), theRot(frame.rotation().transposed()), theTrivial(frame.trivial()) {}
    
    LocalPoint operator()(const GlobalPoint& gp) const {
         return toLocal(gp);
    }

    LocalVector operator()(const GlobalVector& gv) const {
       	 return	toLocal(gv);
    }

    LocalPoint toLocal( const GlobalPoint& gp) const {
      if ((theTrivial)) return LocalPoint(gp.basicVector());
      return LocalPoint( theRot.multiplyInverse( gp.basicVector() -
			 thePos.basicVector()) 
                       );
    }
    
    LocalVector toLocal( const GlobalVector& gv) const {
      if ((theTrivial)) return LocalVector(gv.basicVector());
      return LocalVector(theRot.multiplyInverse(gv.basicVector()));
    } 
    
  // private:
    PositionType  thePos;
    RotationType  theRot;
    bool theTrivial;
    
  };

  

  /** Transform a local point (i.e. a point with coordinates in the
   *  local frame) to the global frame
   */
  GlobalPoint toGlobal( const LocalPoint& lp) const {
    if ((theTrivial)) return GlobalPoint(lp.basicVector());
    return GlobalPoint( rotation().multiplyInverse( lp.basicVector()) +
			position().basicVector());
  }

  /** Transform a local point with different float precision from the
   *  one of the reference frame, and return a global point with the
   *  same precision as the input one.
   */
  template <class U>
  Point3DBase< U, GlobalTag>
  toGlobal( const Point3DBase< U, LocalTag>& lp) const {
    if ((theTrivial)) return Point3DBase< U, GlobalTag>(lp.basicVector());
    return Point3DBase< U, GlobalTag>( rotation().multiplyInverse( lp.basicVector()) +
				       position().basicVector());
  }

  /** Transform a local vector (i.e. a vector with coordinates in the
   *  local frame) to the global frame
   */
  GlobalVector toGlobal( const LocalVector& lv) const {
    if ((theTrivial)) return GlobalVector(lv.basicVector());
    return GlobalVector( rotation().multiplyInverse( lv.basicVector()));
  }

  /** Transform a local vector with different float precision from the
   *  one of the reference frame, and return a global vector with the
   *  same precision as the input one.
   */
  template <class U>
  Vector3DBase< U, GlobalTag>
  toGlobal( const Vector3DBase< U, LocalTag>& lv) const {
    if ((theTrivial)) return Vector3DBase< U, GlobalTag>(lv.basicVector());
    return Vector3DBase< U, GlobalTag>( rotation().multiplyInverse( lv.basicVector()));
  }

  /** Transform a global point (i.e. a point with coordinates in the
   *  global frame) to the local frame
   */
  LocalPoint toLocal( const GlobalPoint& gp) const {
    if ((theTrivial)) return LocalPoint(gp.basicVector());
    return LocalPoint( rotation() * (gp.basicVector()-position().basicVector()));
  }

  /** Transform a global point with different float precision from the
   *  one of the reference frame, and return a local point with the
   *  same precision as the input one.
   */
  template <class U>
  Point3DBase< U, LocalTag>
  toLocal( const Point3DBase< U, GlobalTag>& gp) const {
    if ((theTrivial)) return Point3DBase< U, LocalTag>(gp.basicVector());
    return Point3DBase< U, LocalTag>( rotation() * 
				      (gp.basicVector()-position().basicVector()));
  }

  /** Transform a global vector (i.e. a vector with coordinates in the
   *  global frame) to the local frame
   */
  LocalVector toLocal( const GlobalVector& gv) const {
    if ((theTrivial)) return LocalVector(gv.basicVector());
    return LocalVector( rotation() * gv.basicVector());
  }

  /** Transform a global vector with different float precision from the
   *  one of the reference frame, and return a local vector with the
   *  same precision as the input one.
   */
  template <class U>
  Vector3DBase< U, LocalTag>
  toLocal( const Vector3DBase< U, GlobalTag>& gv) const {
    if ((theTrivial)) return Vector3DBase< U, LocalTag>(gv.basicVector());
    return Vector3DBase< U, LocalTag>( rotation() * gv.basicVector());
  }

  /** Move the position of the frame in the global frame.  
   *  Useful e.g. for alignment.
   */
  void move( const GlobalVector& displacement) {
    thePos += displacement;
    setCache();
  }

  /** Rotate the frame in the global frame.
   *  Useful e.g. for alignment.
   */
  void rotate( const RotationType& rotation) {
    theRot *= rotation;
    setCache();
  }

  bool trivial() const{ return theTrivial;}

private:

  PositionType  thePos;
  RotationType  theRot;

  /*
  void resetCache() {
    if ((thePos.x() == 0.) && (thePos.y() == 0.)) {
      thePhi = theEta = 0.; // avoid FPE
    } else {
      thePhi = iniPhi();
      theEta = iniEta();
    }
  }
 */

  void setCache() {
    setTrivial();
    theDV = rotation().z().dot(-position().basicVector());
    if ((thePos.x() == 0.) & (thePos.y() == 0.)) {
      thePhi = theEta = 0.; // avoid FPE
    } else {
      thePhi = thePos.barePhi();
      theEta = thePos.eta();
    }
  }
  
  void  setTrivial() {
    theTrivial =   
      thePos == PositionType(0,0,0)
      && theRot.xx() == T(1)
      && theRot.yy() ==	T(1)
      && theRot.zz() ==	T(1)
      ;

  }

  T thePhi;
  T theEta;
  T theDV;  // normal distance of origin
  bool theTrivial; 

};
  
#endif
