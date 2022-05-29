#pragma once
#include "HeterogeneousCore/CUDAUtilities/interface/Buffer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/Queue.h"

#include <vector>

#include <cassert>

namespace memoryPool {

    void initCPU();
    void shutdownCPU();

    void initCUDA();
    void shutdownCUDA();
    void dumpStatCUDA();


    SimplePoolAllocator *getPool(Queue queue, Where where) {
      return  Where::onDevice == where ? queue->m_devicePool : queue->m_hostPool;
    }

    // allocate either on current device or host
    std::pair<void *, int> alloc(uint64_t size, SimplePoolAllocator &pool);

    // schedule free
    void free(Queue queue, std::vector<int> buckets, SimplePoolAllocator &pool);

    // copy fanout
    void doCopy(void * dst,  void const * src,  uint64_t size, Queue queue);


    template <typename T>
    void copy(Buffer<T> &dst, Buffer<T> const &src, Queue queue, uint64_t size) {
      assert(dst.get());
      assert(src.get());
      assert(size > 0);
      doCopy(dst.get(),src.get(),size,queue);
    }
    template <typename T>
    void copy(Buffer<T> &dst, Buffer<T> const &src, Queue queue) {
      copy(dst,src,queue,src.size());
    }

    struct DeviceDeleterBase : public DeleterBase {
      DeviceDeleterBase(Queue queue, Where where) : DeleterBase(getPool(queue, where)), m_queue(std::move(queue)) {}

      DeviceDeleterBase(Queue const &queue, SimplePoolAllocator *pool) : DeleterBase(pool), m_queue(std::move(queue)) {}

      ~DeviceDeleterBase() override = default;

      Queue m_queue;
    };

    struct DeleteOne final : public DeviceDeleterBase {
      using DeviceDeleterBase::DeviceDeleterBase;

      ~DeleteOne() override = default;
      void operator()(int bucket) override { free(m_queue, std::vector<int>(1, bucket), *pool()); }
    };

    struct BundleDelete final : public DeviceDeleterBase {
      BundleDelete(Queue const &queue, Where where) : DeviceDeleterBase(queue, where) { m_buckets.reserve(8); }

      ~BundleDelete() override { free(m_queue, std::move(m_buckets), *pool()); }

      void operator()(int bucket) override { m_buckets.push_back(bucket); }

      std::vector<int> m_buckets;
    };

    template <typename T>
    Buffer<T> makeBuffer(uint64_t size, Deleter del) {
      auto ret = alloc(sizeof(T) * size, *del.pool());
      if (ret.second < 0) {
        std::cout << "could not allocate " << size << ' ' << typeid(T).name() << " of size " << sizeof(T) << std::endl;
        throw std::bad_alloc();
      }
      return Buffer<T>((T *)(ret.first), ret.second, size, std::move(del));
    }

    template <typename T>
    Buffer<T> makeBuffer(uint64_t size, Queue const &queue, Where where) {
      return makeBuffer<T>(size, Deleter(std::make_shared<DeleteOne>(queue, where)));
    }

}  // namespace memoryPool
