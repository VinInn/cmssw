#ifndef Grid3D_h
#define Grid3D_h

/** \class Grid3D
 *
 *  Implementation of a 3D regular grid.
 *
*  \author T. Todorov
 */

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
// #include "DataFormats/Math/interface/SIMDVec.h"
#include "Grid1D.h"
#include <vector>
#include "FWCore/Utilities/interface/Visibility.h"

#include "zfp-0.5.0/interface/zfparray3.h"

// the storage class
// needed just because legacy software used () constructor
struct BStorageArray {
  BStorageArray(){}
  BStorageArray(float x,float y, float z) : v{x,y,z}{}

  float const & operator[](int i) const { return v[i];}

  float v[3];
};

class dso_internal Grid3D {
public:

 // typedef double   Scalar;
  typedef float   Scalar;
  typedef Basic3DVector<Scalar>   ValueType;
  typedef ValueType ReturnType; 
 
  using BVector = BStorageArray;
  //using BVector =  ValueType;
  using Container = std::vector<BVector>;

  using ZipArray = zfp::array3<float>;

  Grid3D() {}

  Grid3D( const Grid1D& ga, const Grid1D& gb, const Grid1D& gc,
	  std::vector<BVector>& data);


  //  Grid3D( const Grid1D& ga, const Grid1D& gb, const Grid1D& gc,
  //	  std::vector<ValueType> const & data);


  int index(int i, int j, int k) const {return i*stride1_ + j*stride2_ + k;}
  int stride1() const { return stride1_;}
  int stride2() const { return stride2_;}
  int stride3() const { return 1;}
  ValueType operator()(int i) const {
    return ValueType(data_[i][0],data_[i][1],data_[i][2]);
  }

  ValueType operator()(int i, int j, int k) const {
    return ValueType(zipX(k,j,i),zipY(k,j,i),zipZ(k,j,i));
   //  return (*this)(index(i,j,k));
  }

  const Grid1D& grida() const {return grida_;}
  const Grid1D& gridb() const {return gridb_;}
  const Grid1D& gridc() const {return gridc_;}

  const Container & data() const {return data_;}

  void dump() const;

private:

  Grid1D grida_;
  Grid1D gridb_;
  Grid1D gridc_;


  ZipArray zipX,zipY,zipZ;
  Container data_;

  int stride1_;
  int stride2_;


};

#endif
