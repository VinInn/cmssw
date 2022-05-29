#include "HeterogeneousCore/CUDAUtilities/interface/Queue.h"

namespace memoryPool {

  Queue getCPUQueue();
  Queue getCudaQueue(void *);
  Queue getHipQueue(void *);

  Queue getQueue(void * stream, DeviceAPI api) {
     switch (api) {
        case DeviceAPI::posix :
          return getCPUQueue();
        case DeviceAPI::cuda :
          return getCudaQueue(stream);
        case DeviceAPI::hip :
          return getHipQueue(stream);
        default :
          return Queue();
     }
  }

}


#include "HeterogeneousCore/CUDAUtilities/interface/SimplePoolAllocator.h"


namespace memoryPool {
    // allocate either on current device or on host
    std::pair<void *, int> alloc(uint64_t size, SimplePoolAllocator &pool) {
      int i = pool.alloc(size);
      void *p = pool.pointer(i);
      return std::pair<void *, int>(p, i);
    }

    // schedule free
    void free(Queue queue, std::vector<int> buckets, SimplePoolAllocator &pool) {
      auto payload = new Payload{&pool, std::move(buckets)};
      pool.scheduleFree(payload, queue->m_stream);
    }


   // alternative are virtual functions in Queue
    void cpuCopy(void * dst,  void const * src,  uint64_t size);
    void cudaCopy(void *    dst,  void const * src,     uint64_t size, void *);
    void hipCopy(void *    dst,  void const * src,     uint64_t size, void *);

   void doCopy(void * dst,  void const * src,  uint64_t size, Queue queue) {
       switch (queue->m_api) {
        case DeviceAPI::posix :
          cpuCopy(dst,src,size); break;
        case DeviceAPI::cuda :
          cudaCopy(dst,src,size, queue->m_stream); break;
        case DeviceAPI::hip :
             hipCopy(dst,src,size, queue->m_stream); break;
        default :
         break;
      }
    }


}

