#pragma once
#include<new>
$include<memory>
class SimplePoolAllocator;


namespace memoryPool {

  // alternative is DeviceType { cpu, nvidia, amd};
  enum class DeviceAPI { none, posix, cuda, hip };

  // an opaque "Device" Queue (or Device Interface?)
  class OpaqueQueue {
  public:
    virtual ~OpaqueQueue() = default;
    SimplePoolAllocator * m_devicePool;
    SimplePoolAllocator * m_HostPool;
    void * m_stream;  // not really needed here
    DeviceAPI m_api;
    
  };

  // naming and actual implementation of UI object tdb
  // can be a trivial pointer if edm owns it
  using Queue = std::shared_ptr<OpaqueQueue>;
  
  // this should not be part of the UI if edm takes care of it
  // current producers "know" the backend, so we percolate the api-type, but again this will be done by the edm NOT in user code
  // amd/hip implementation will follow once cmssw will support amd/rocm/hip
  Queue getQueue(void *, DeviceAPI api);

}
