#ifndef HeterogeneousCore_CUDAUtilities_memsetAsync_h
#define HeterogeneousCore_CUDAUtilities_memsetAsync_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cpu_unique_ptr.h"

#include <type_traits>
#include <cstring>

namespace cudautils {
  template <typename T>
  inline void memsetAsync(cudautils::device::unique_ptr<T>& ptr, T value, cudaStream_t stream) {
    // Shouldn't compile for array types because of sizeof(T), but
    // let's add an assert with a more helpful message
    static_assert(std::is_array<T>::value == false, "For array types, use the other overload with the size parameter");
    cudaCheck(cudaMemsetAsync(ptr.get(), value, sizeof(T), stream));
  }

  template <typename T>
  inline void memsetAsync(cudautils::cpu::unique_ptr<T>& ptr, T value, cudaStream_t) {
    ::memset(ptr.get(), value, sizeof(T));
  }

  /**
   * The type of `value` is `int` because of `cudaMemsetAsync()` takes
   * it as an `int`. Note that `cudaMemsetAsync()` sets the value of
   * each **byte** to `value`. This may lead to unexpected results if
   * `sizeof(T) > 1` and `value != 0`.
   */
  template <typename T>
  inline void memsetAsync(cudautils::device::unique_ptr<T[]>& ptr, int value, size_t nelements, cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(ptr.get(), value, nelements * sizeof(T), stream));
  }
  template <typename T>
  inline void memsetAsync(cudautils::cpu::unique_ptr<T[]>& ptr, int value, size_t nelements, cudaStream_t) {
    ::memset(ptr.get(), value, nelements * sizeof(T));
  }


}  // namespace cudautils

#endif
