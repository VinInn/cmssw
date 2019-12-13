#ifndef HeterogeneousCore_CUDATest_CUDAThing_H
#define HeterogeneousCore_CUDATest_CUDAThing_H

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

class CUDAThing {
public:
  struct View {
    float *eta, *phi, *r;
    int nHits;
  };


  auto & view() { return m_view;}
  auto const & view() const { return m_view;}

  auto nHIts() const {return m_view.nHits;}

  CUDAThing() = default;
  CUDAThing(cudautils::device::unique_ptr<float[]> ptr) : ptr_(std::move(ptr)) {}
  explicit CUDAThing(int iNHits, cudaStream_t stream) {
    ptr_ = cudautils::make_device_unique<float[]>(3*iNHits, stream);
    m_view.nHits=iNHits;
    m_view.eta = ptr_.get();
    m_view.phi = ptr_.get()+iNHits;
    m_view.r = ptr_.get()+2*iNHits;
  }

  const float *get() const { return ptr_.get(); }

private:
  cudautils::device::unique_ptr<float[]> ptr_;
  View m_view;
};

#endif
