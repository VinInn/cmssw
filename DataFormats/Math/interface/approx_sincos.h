#ifndef DataFormatsMathAPPROX_SINCOS_H
#define DataFormatsMathAPPROX_SINCOS_H

#include <cmath>
#include <tuple>

// only for  -45deg < x < 45deg
template <typename Float>
inline std::tuple<Float, Float> approx_sincosf0(Float x) {
  auto z = x * x;

  auto s = (((-1.9515295891E-4f * z + 8.3321608736E-3f) * z - 1.6666654611E-1f) * z * x) + x;

  auto c =
      ((2.443315711809948E-005f * z - 1.388731625493765E-003f) * z + 4.166664568298827E-002f) * z * z - 0.5f * z + 1.0f;

  return {s, c};
}

// valid only in -pi < x < pi
// 2x faster than vdt, worse case: 3 times less precise
template <typename Float>
inline std::tuple<Float, Float> approx_sincos(Float x) {
  // reduce to "first quadrant"
  auto xx = 0.25f * x;
  auto [ls, lc] = approx_sincosf0(xx);

  // back...
  // use poly expansion
  // http://mathworld.wolfram.com/Multiple-AngleFormulas.html
  auto ss = ls * ls;
  auto s = 8.f * lc * ls * (0.5f - ss);
  auto c = 1.f - 8.f * ss * (1.0f - ss);

  return {s, c};
}

// valid only in -pi < x < pi
template <typename Float>
inline Float approx_sin(Float x) {
  auto [s, c] = approx_sincos(x);
  return s;
}

// valid only in -pi < x < pi
template <typename Float>
inline Float approx_cos(Float x) {
  auto [s, c] = approx_sincos(x);
  approx_sincos(x);
  return c;
}

#endif
