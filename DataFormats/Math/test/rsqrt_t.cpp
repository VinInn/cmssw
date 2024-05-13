#include "DataFormats/Math/interface/rsqrt.h"
#include "DataFormats/Math/interface/FastMath.h"

#include<iostream>
#include<cmath>




int main(int argn, char**) {

  float f = 1. + argn;
  double d = f;
  __float128 dd = d;

  std::cout << std::hexfloat << f << ' ' << 1.f/std::sqrt(f)  << ' ' << approx_math::rsqrt(f) << ' ' << vdt::fast_isqrtf_general(f,1) << ' ' << vdt::fast_isqrtf_general(f,2) <<  ' ' << fastmath::invSqrt(f) << std::endl;
  std::cout << std::hexfloat << d << ' ' << 1./std::sqrt(d)   << ' ' << approx_math::rsqrt(d) << ' ' << vdt::fast_isqrt_general(d,2) << ' ' << vdt::fast_isqrt_general(d,3) << ' ' << vdt::fast_isqrt_general(d,4) << std::endl;
  std::cout << std::hexfloat << (double)dd << ' ' << 1./std::sqrt((double)(dd))  << ' ' << (double)approx_math::rsqrt(dd) << std::endl;

}
