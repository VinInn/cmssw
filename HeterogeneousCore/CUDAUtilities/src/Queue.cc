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
        case default :
             break;
     }
  }

}



