#ifndef DataFormatsMathAPPROX_SINCOS_H
#define DataFormatsMathAPPROX_SINCOS_H

#include <cmath>

// only for  -45deg < x < 45deg
template<typename Float>
inline void approx_sincosf0(Float x, Float & s, Float &c ) {

  // constexpr float kk = (M_PI/8)*(M_PI/8); 
  auto z = x * x;
  
  s = (((-1.9515295891E-4f * z
	 + 8.3321608736E-3f) * z
	- 1.6666654611E-1f) * z * x)
    + x;
  
  
  c = ((  2.443315711809948E-005f * z
	  - 1.388731625493765E-003f) * z
       + 4.166664568298827E-002f) * z * z
    - 0.5f * z + 1.0f;
  
}


// valid only in -pi < x < pi
// 1.5 faster than vdt, worse case: 3 times less precise
template<typename Float>
inline void approx_sincos( Float x, Float & s, Float &c ) {

  // reduce to "first quadrant"
  auto xx = 0.25f*x;
  Float ls,lc;

  approx_sincosf0(xx,ls,lc);

  // back...
  // use poly expansion
  // http://mathworld.wolfram.com/Multiple-AngleFormulas.html
  auto ss= ls*ls;
  s = 8.f*lc*ls*(0.5f-ss);
  c = 1.f-8.f*ss*(1.0f-ss);

}


// valid only in -pi < x < pi
template<typename Float>
inline Float approx_sin(Float x) {
 Float  s, c;
 approx_sincos(x,s,c); 
 return s;
}

// valid only in -pi < x < pi
template<typename Float>
inline Float approx_cos(Float x) {
 Float  s, c;
 approx_sincos(x,s,c);
 return c;
}    

#endif
