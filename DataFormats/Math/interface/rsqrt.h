#ifndef DataFormatsMathRSQRT_H
#define DataFormatsMathRSQRT_H

#include <vdt/sqrt.h>

namespace approx_math {

// fast portable rsqrt
template<typename T>
T rsqrt(T x) {
  if constexpr (std::is_same<T, float>::value)
    return vdt::fast_isqrtf_general(x,2);
  if constexpr (std::is_same<T, double>::value)
    return vdt::fast_isqrt_general(x,3);
  return vdt::fast_isqrt_general((double)x,4); 
}

}



#endif // DataFormatsMathRSQRT_H
