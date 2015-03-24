#ifndef DataFormatsGeometryVectorHessianPlan_H
#define	DataFormatsGeometryVectorHessianPlan_H

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"


/**
 * Hessian Normal form of a plane representation
 */
template<typename T>
class HessianPlane {
public:
  // typedef typename Basic3DVector<T>::MathVector MathVector;
  HessianPlane(){}
  // constructor from normal and distance of origin
  HessianPlane(Basic3DVector<T> n, T d) : m_me(n)  {m_me[3]=d;}

  /// signed distance from plane
  T localZ(Basic3DVector<T> p) const {
    p[3]=T(1);
    return m_me.dot(p); // +m_dv;
  }

  Basic3DVector<T> basicVector() const { return m_me;}

  T nx() const { return m_me[0];}
  T ny() const { return	m_me[1];}
  T nz() const { return	m_me[2];}
//  T dv() const { return m_dv;}
  T dv() const { return	m_me[3];}


private:

  Basic3DVector<T> m_me;
//  T m_dv;

};



#endif
