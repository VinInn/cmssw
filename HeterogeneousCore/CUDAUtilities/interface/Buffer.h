#pragma once
#include <memory>
#include <new>

#include <iostream>
class SimplePoolAllocator;

namespace memoryPool {

  enum class Where { none, onDevice, onHost, unified };

  class DeleterBase {
  public:
    explicit DeleterBase(SimplePoolAllocator* pool) : m_pool(pool) {}
    virtual ~DeleterBase() = default;
    virtual void operator()(int bucket) = 0;

    SimplePoolAllocator* pool() const { return m_pool; }

  protected:
    SimplePoolAllocator* m_pool;
  };

  class Deleter {
  public:
    Deleter() = default;
    explicit Deleter(std::shared_ptr<DeleterBase> del) : me(std::move(del)) {}

    void set(std::shared_ptr<DeleterBase> const& del) { me = del; }
    std::shared_ptr<DeleterBase> const& get() const { return me; }

    void operator()(int bucket) {
      if (!me) {
        // other possibility is to assume it has been created by a streamer with "new" so we just call delete[] (and we need the pointer here)
        std::cout << "deleter w/o implementation!!!" << std::endl;
        throw std::bad_alloc();
      }
      if (bucket < 0)
        // can be used also to mark those allocated directly (in this case the deleter is required)  (and we need to pass the pointer as well) 
        std::cout << "delete with negative bucket!!!" << std::endl;
      (*me)(bucket);
    }

    SimplePoolAllocator* pool() const { return me->pool(); }

  private:
    std::shared_ptr<DeleterBase> me;  //!
  };

  template <typename T>
  class Buffer {
  public:
    typedef T value_type;
    typedef T* pointer;
    typedef T& reference;
    typedef T const* const_pointer;
    typedef T const& const_reference;

    struct Data { T * p; uint64_t size; int bucket;};

    Buffer() = default;
    Buffer(T* p, int bucket, uint64_t size) : m_p(p), m_size(size), m_bucket(bucket) {}
    Buffer(T* p, int bucket, uint64_t size, Deleter del) : m_deleter(std::move(del)), m_p(p), m_size(size), m_bucket(bucket) {}
    Buffer(Data const& rh, Deleter const& del) : m_deleter(del), m_p(rh.p), m_size(rh.size), m_bucket(rh.bucket) {}
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;

    template <typename U>
    Buffer(Buffer<U>&& rh) : Buffer(rh.release(), rh.size(), rh.deleter()) {}
    template <typename U>
    Buffer& operator=(Buffer<U>&& rh) {
      reset(rh.release());
      m_deleter = rh.deleter();
      return *this;
    }

    ~Buffer() {
      // assert(m_p == pool()->pointer(m_bucket));  // does not compile with current dependency of headers.
      if (m_p)
        m_deleter(m_bucket);
    }

    auto size() const { return m_size;}

    pointer get() { return m_p; }
    const_pointer get() const { return m_p; }
    reference operator*() { return *m_p; }
    const_reference operator*() const { return *m_p; }
    pointer operator->() { return get(); }
    const_pointer operator->() const { return get(); }
    reference operator[](int i) { return m_p[i]; }
    const_reference operator[](int i) const { return m_p[i]; }

    Deleter& deleter() { return m_deleter; }
    Deleter const& deleter() const { return m_deleter; }
    SimplePoolAllocator* pool() const { return deleter().pool(); }

    int bucket() const { return m_bucket; }

      Data  release() {
      Data ret{m_p, m_size, m_bucket};
      m_p = nullptr;
      m_size = 0;
      m_bucket = -1;
      return ret;
    }

    void reset() {
      if (m_p)
        m_deleter(m_bucket);
      m_p = nullptr;
      m_bucket = -1;
      m_size=0;
    }

    void reset(Data const& rh) {
      if (m_p)
        m_deleter(m_bucket);
      m_p = rh.p;
      m_size = rh.size;
      m_bucket = rh.bucket;
    }

  private:
    Deleter m_deleter;      //!
    pointer m_p = nullptr;  //!
    uint64_t m_size = 0;    //!
    int m_bucket = -1;      //!
  };

}  // namespace memoryPool
