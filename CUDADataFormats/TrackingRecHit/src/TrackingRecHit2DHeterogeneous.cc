#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DSOAView.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

#include "HeterogeneousCore/CUDAUtilities/interface/memoryPool.h"

TrackingRecHit2DHeterogeneous::TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                                             bool isPhase2,
                                                             int32_t offsetBPIX2,
                                                             pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                             uint32_t const* hitsModuleStart,
                                                             memoryPool::Queue queue,
                                                             TrackingRecHit2DHeterogeneous const* input)
    : m_nHits(nHits), m_offsetBPIX2(offsetBPIX2), m_hitsModuleStart(hitsModuleStart) {

  auto where = input == nullptr ? memoryPool::Where::onDevice : memoryPool::Where::onHost;

  if (input) assert(nHits==input->nHits());

  memoryPool::Deleter deleter = memoryPool::Deleter(std::make_shared<memoryPool::BundleDelete>(queue, where));
  assert(deleter.pool());

  auto view = memoryPool::makeBuffer<TrackingRecHit2DSOAView>(1, queue, memoryPool::Where::onHost);
  assert(view.get());
  m_nMaxModules = isPhase2 ? phase2PixelTopology::numberOfModules : phase1PixelTopology::numberOfModules;
  assert(view.get());
  view->m_nHits = nHits;
  view->m_nMaxModules = m_nMaxModules;

  m_view = memoryPool::makeBuffer<TrackingRecHit2DSOAView>(1, deleter);  // leave it on host and pass it by value?
  assert(m_view.get());
  m_AverageGeometryStore = memoryPool::makeBuffer<TrackingRecHit2DSOAView::AverageGeometry>(1, deleter);

  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) {
    if (memoryPool::Where::onDevice == where) {
      memoryPool::copy(m_view, view, queue);
    } else {
      m_view.reset(view.release());
    }
    return;
  }

  // the single arrays are not 128 bit alligned...
  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated

  // host copy is "reduced"  (to be reviewed at some point)
  if (memoryPool::Where::onHost == where) {
    // it has to compile for ALL cases
    copyFromGPU(input, queue);
  } else {
    assert(input == nullptr);

    auto nL = isPhase2 ? phase2PixelTopology::numberOfLayers : phase1PixelTopology::numberOfLayers;

    m_store16 = memoryPool::makeBuffer<uint16_t>(nHits * n16, deleter);
    m_store32 = memoryPool::makeBuffer<float>(nHits * n32 + nL + 1, deleter);
    m_PhiBinnerStore = memoryPool::makeBuffer<TrackingRecHit2DSOAView::PhiBinner>(1, deleter);
  }

  
  static_assert(sizeof(TrackingRecHit2DSOAView::hindex_type) == sizeof(float));
  static_assert(sizeof(TrackingRecHit2DSOAView::hindex_type) == sizeof(TrackingRecHit2DSOAView::PhiBinner::index_type));

  auto get32 = [&](Storage32 i) { return m_store32.get() + static_cast<int>(i) * nHits; };

  // copy all the pointers
  m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
  m_phiBinnerStorage = view->m_phiBinnerStorage =
      reinterpret_cast<TrackingRecHit2DSOAView::PhiBinner::index_type*>(get32(Storage32::kPhiStorage));

  view->m_xl = get32(Storage32::kXLocal);
  view->m_yl = get32(Storage32::kYLocal);
  view->m_xerr = get32(Storage32::kXerror);
  view->m_yerr = get32(Storage32::kYerror);
  view->m_chargeAndStatus = reinterpret_cast<uint32_t*>(get32(Storage32::kCharge));

  if (memoryPool::Where::onHost != where) {
    assert(input == nullptr);
    view->m_xg = get32(Storage32::kXGlobal);
    view->m_yg = get32(Storage32::kYGlobal);
    view->m_zg = get32(Storage32::kZGlobal);
    view->m_rg = get32(Storage32::kRGlobal);

    auto get16 = [&](Storage16 i) { return m_store16.get() + static_cast<int>(i) * nHits; };
    m_iphi = view->m_iphi = reinterpret_cast<int16_t*>(get16(Storage16::kPhi));

    view->m_xsize = reinterpret_cast<int16_t*>(get16(Storage16::kXSize));
    view->m_ysize = reinterpret_cast<int16_t*>(get16(Storage16::kYSize));
    view->m_detInd = get16(Storage16::kDetId);

    m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
    m_hitsLayerStart = view->m_hitsLayerStart = reinterpret_cast<uint32_t*>(get32(Storage32::kLayers));
  }

  // transfer view
  if (memoryPool::Where::onDevice == where) {
    memoryPool::copy(m_view, view, queue);
  } else {
    m_view.reset(view.release());
  }
}

//this is intended to be used only for CPU
TrackingRecHit2DHeterogeneous::TrackingRecHit2DHeterogeneous(
    float* store32, uint16_t* store16, uint32_t* modules, int nHits, memoryPool::Queue queue)
    : m_nHits(nHits), m_hitsModuleStart(modules) {

  assert(nullptr==queue->m_stream);
  assert(queue->m_api==memoryPool::DeviceAPI::posix);
  auto where = memoryPool::Where::onDevice;

  m_view = memoryPool::makeBuffer<TrackingRecHit2DSOAView>(1, queue, where);
  auto view = m_view.get();
  assert(view);
  view->m_nHits = nHits;

  if (0 == nHits) {
    return;
  }

  m_store16 = memoryPool::makeBuffer<uint16_t>(nHits * n16, queue, where);
  m_store32 = memoryPool::makeBuffer<float>(nHits * n32, queue, where);
  m_PhiBinnerStore = memoryPool::makeBuffer<TrackingRecHit2DSOAView::PhiBinner>(1, queue, where);
  m_AverageGeometryStore = memoryPool::makeBuffer<TrackingRecHit2DSOAView::AverageGeometry>(1, queue, where);

  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_hitsModuleStart = m_hitsModuleStart;

  //store transfer
    std::copy(store32, store32 + nHits * n32, m_store32.get());  // want to copy it
    std::copy(store16, store16 + nHits * n16, m_store16.get());

  //getters
  auto get32 = [&](Storage32 i) { return m_store32.get() + static_cast<int>(i) * nHits; };
  auto get16 = [&](Storage16 i) { return m_store16.get() + static_cast<int>(i) * nHits; };

  //Store 32
  view->m_xl = get32(Storage32::kXLocal);
  view->m_yl = get32(Storage32::kYLocal);
  view->m_xerr = get32(Storage32::kXerror);
  view->m_yerr = get32(Storage32::kYerror);
  view->m_chargeAndStatus = reinterpret_cast<uint32_t*>(get32(Storage32::kCharge));
  view->m_xg = get32(Storage32::kXGlobal);
  view->m_yg = get32(Storage32::kYGlobal);
  view->m_zg = get32(Storage32::kZGlobal);
  view->m_rg = get32(Storage32::kRGlobal);

  m_phiBinner = view->m_phiBinner = m_PhiBinnerStore.get();
  m_phiBinnerStorage = view->m_phiBinnerStorage =
      reinterpret_cast<TrackingRecHit2DSOAView::PhiBinner::index_type*>(get32(Storage32::kPhiStorage));

  //Store 16
  view->m_detInd = get16(Storage16::kDetId);
  m_iphi = view->m_iphi = reinterpret_cast<int16_t*>(get16(Storage16::kPhi));
  view->m_xsize = reinterpret_cast<int16_t*>(get16(Storage16::kXSize));
  view->m_ysize = reinterpret_cast<int16_t*>(get16(Storage16::kYSize));

}

using namespace memoryPool;

memoryPool::Buffer<float> TrackingRecHit2DGPU::localCoordToHostAsync(Queue queue) const {
  auto ret = memoryPool::makeBuffer<float>(5 * nHits(), queue, memoryPool::Where::onHost);
  memoryPool::copy(ret, m_store32, 5 * nHits(), queue);
  return ret;
}

memoryPool::Buffer<float> TrackingRecHit2DGPU::store32ToHostAsync(Queue queue) const {
  int nElem = nHits() * n32;
  auto ret = memoryPool::makeBuffer<float>(nElem, queue, memoryPool::Where::onHost);
  memoryPool::copy(ret, m_store32, nElem, queue);
  return ret;
}


memoryPool::Buffer<uint16_t> TrackingRecHit2DGPU::store16ToHostAsync(Queue queue) const {
  auto ret = memoryPool::makeBuffer<uint16_t>(m_store16.size(), queue, memoryPool::Where::onHost);
  memoryPool::copy(ret, m_store16, queue);
  return ret;
}

memoryPool::Buffer<uint32_t> TrackingRecHit2DGPU::hitsModuleStartToHostAsync(Queue queue) const {
  auto ret = memoryPool::makeBuffer<uint32_t>(nMaxModules() + 1, queue, memoryPool::Where::onHost);
  if (m_hitsModuleStart)
    memoryPool::copyToHost(ret.get(), m_hitsModuleStart, sizeof(uint32_t)*ret.size() ,queue); 
  return ret;
}

void TrackingRecHit2DHost::copyFromGPU(TrackingRecHit2DGPU const* input, Queue queue) {
  assert(input);
  m_store32 = input->localCoordToHostAsync(queue);
}
