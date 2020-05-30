#ifndef HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
#define HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h

// #define GPU_DEBUG

#include <algorithm>
#ifndef __CUDA_ARCH__
#include <atomic>
#endif  // __CUDA_ARCH__
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "HeterogeneousCore/CUDAUtilities/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"

#include "HeterogeneousCore/CUDAUtilities/interface/prefixScan.h"

namespace cms {
  namespace cuda {

    template <typename Histo, typename T>
    __global__ void countFromVector(Histo *__restrict__ h,
                                    uint32_t nh,
                                    T const *__restrict__ v,
                                    uint32_t const *__restrict__ offsets) {
      int first = blockDim.x * blockIdx.x + threadIdx.x;
      for (int i = first, nt = offsets[nh]; i < nt; i += gridDim.x * blockDim.x) {
        auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
        assert((*off) > 0);
        int32_t ih = off - offsets - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        (*h).count(v[i], ih);
      }
    }

    template <typename Histo, typename T>
    __global__ void fillFromVector(Histo *__restrict__ h,
                                   uint32_t nh,
                                   T const *__restrict__ v,
                                   uint32_t const *__restrict__ offsets) {
      int first = blockDim.x * blockIdx.x + threadIdx.x;
      for (int i = first, nt = offsets[nh]; i < nt; i += gridDim.x * blockDim.x) {
        auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
        assert((*off) > 0);
        int32_t ih = off - offsets - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        (*h).fill(v[i], i, ih);
      }
    }

    template <typename Histo>
    inline __attribute__((always_inline)) void launchZero(Histo *__restrict__ h,
                                                          cudaStream_t stream
#ifndef __CUDACC__
                                                          = cudaStreamDefault
#endif
    ) {
      uint32_t *poff = (uint32_t *)((char *)(h) + offsetof(Histo, off));
      int32_t size = offsetof(Histo, bins) - offsetof(Histo, off);
      assert(size >= int(sizeof(uint32_t) * Histo::totbins()));
#ifdef __CUDACC__
      cudaCheck(cudaMemsetAsync(poff, 0, size, stream));
#else
      ::memset(poff, 0, size);
#endif
    }

    template <typename Histo>
    inline __attribute__((always_inline)) void launchFinalize(Histo *__restrict__ h,
                                                              cudaStream_t stream
#ifndef __CUDACC__
                                                              = cudaStreamDefault
#endif
    ) {
#ifdef __CUDACC__
      uint32_t *poff = (uint32_t *)((char *)(h) + offsetof(Histo, off));
      CUDATask *ptasks = (CUDATask *)((char *)(h) + offsetof(Histo, tasks));
      uint32_t *ppsws = (uint32_t *)((char *)(h) + offsetof(Histo, psws));

      auto nthreads = Histo::nthreads();
      auto nblocks = Histo::nblocks();
      multiTaskPrefixScanKernel<<<nblocks, nthreads, 0, stream>>>(poff, poff, Histo::totbins(), ptasks, ppsws);
      cudaCheck(cudaGetLastError());
#else
      h->finalize();
#endif
    }

    template <typename Histo, typename T>
    __global__ void fillManyFromVectorKernel(Histo *__restrict__ ph,
                                             uint32_t nh,
                                             T const *__restrict__ v,
                                             uint32_t const *__restrict__ offsets) {
      auto &h = *ph;
      auto count = [&](int32_t iWork) {
        int first = blockDim.x * iWork + threadIdx.x;
        for (int i = first, nt = offsets[nh]; i < nt; i += gridDim.x * blockDim.x) {
          auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
          assert((*off) > 0);
          int32_t ih = off - offsets - 1;
          assert(ih >= 0);
          assert(ih < int(nh));
          h.count(v[i], ih);
        }
      };

      auto fill = [&](int32_t iWork) {
        int first = blockDim.x * iWork + threadIdx.x;
        for (int i = first, nt = offsets[nh]; i < nt; i += gridDim.x * blockDim.x) {
          auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
          assert((*off) > 0);
          int32_t ih = off - offsets - 1;
          assert(ih >= 0);
          assert(ih < int(nh));
          h.fill(v[i], i, ih);
        }
      };

      h.countAndFill(count, fill);
    }

    template <typename Histo, typename T>
    inline __attribute__((always_inline)) void fillManyFromVector(Histo *__restrict__ h,
                                                                  uint32_t nh,
                                                                  T const *__restrict__ v,
                                                                  uint32_t const *__restrict__ offsets,
                                                                  uint32_t totSize,
                                                                  int nthreads,
                                                                  cudaStream_t stream
#ifndef __CUDACC__
                                                                  = cudaStreamDefault
#endif
    ) {
      launchZero(h, stream);
#ifdef __CUDACC__

#ifdef NO_CUDATASK
      auto nblocks = (totSize + nthreads - 1) / nthreads;
      countFromVector<<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
      cudaCheck(cudaGetLastError());
      launchFinalize(h, stream);
      fillFromVector<<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
      cudaCheck(cudaGetLastError());
#else
      nthreads = Histo::nthreads();
      // keep the number of blocks low
      auto nblocks = std::min(256, int(totSize + nthreads - 1) / nthreads);
      // std::cout << "fillManyFromVectorKernel " << nblocks <<' '<< nthreads << std::endl;
      fillManyFromVectorKernel<<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
      cudaCheck(cudaGetLastError());
#endif
#else
      countFromVector(h, nh, v, offsets);
      h->finalize();
      fillFromVector(h, nh, v, offsets);
#endif
    }

    template <typename Assoc>
    __global__ void finalizeBulk(AtomicPairCounter const *apc, Assoc *__restrict__ assoc) {
      assoc->bulkFinalizeFill(*apc);
    }

    // iteratate over N bins left and right of the one containing "v"
    template <typename Hist, typename V, typename Func>
    __host__ __device__ __forceinline__ void forEachInBins(Hist const &hist, V value, int n, Func func) {
      int bs = Hist::bin(value);
      int be = std::min(int(Hist::nbins() - 1), bs + n);
      bs = std::max(0, bs - n);
      assert(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }

    // iteratate over bins containing all values in window wmin, wmax
    template <typename Hist, typename V, typename Func>
    __host__ __device__ __forceinline__ void forEachInWindow(Hist const &hist, V wmin, V wmax, Func const &func) {
      auto bs = Hist::bin(wmin);
      auto be = Hist::bin(wmax);
      assert(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }

    template <typename T,                  // the type of the discretized input values
              uint32_t NBINS,              // number of bins
              uint32_t SIZE,               // max number of element
              uint32_t S = sizeof(T) * 8,  // number of significant bits in T
              typename I = uint32_t,  // type stored in the container (usually an index in a vector of the input values)
              uint32_t NHISTS = 1     // number of histos stored
              >
    class HistoContainer {
    public:
      using Counter = uint32_t;

      using CountersOnly = HistoContainer<T, NBINS, 0, S, I, NHISTS>;

      using index_type = I;
      using UT = typename std::make_unsigned<T>::type;

      static constexpr uint32_t ilog2(uint32_t v) {
        constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
        constexpr uint32_t s[] = {1, 2, 4, 8, 16};

        uint32_t r = 0;  // result of log2(v) will go here
        for (auto i = 4; i >= 0; i--)
          if (v & b[i]) {
            v >>= s[i];
            r |= s[i];
          }
        return r;
      }

      static constexpr uint32_t sizeT() { return S; }
      static constexpr uint32_t nbins() { return NBINS; }
      static constexpr uint32_t nhists() { return NHISTS; }
      static constexpr uint32_t totbins() { return NHISTS * NBINS + 1; }
      static constexpr uint32_t nbits() { return ilog2(NBINS - 1) + 1; }
      static constexpr uint32_t capacity() { return SIZE; }

      static constexpr int32_t nthreads() { return 256; }
      static constexpr int32_t nblocks() { return (totbins() + nthreads() - 1) / nthreads(); }

      static_assert(nblocks() <= 1024, "too many blocks to perform prefix scan");
      static_assert(nblocks() >= totbins() / nthreads(), "too few blocks to perform prefix scan");

      static constexpr auto histOff(uint32_t nh) { return NBINS * nh; }

      static constexpr UT bin(T t) {
        constexpr uint32_t shift = sizeT() - nbits();
        constexpr uint32_t mask = (1 << nbits()) - 1;
        return (t >> shift) & mask;
      }

      __host__ __device__ void zero() {
        for (auto &i : off)
          i = 0;
      }

      __host__ __device__ __forceinline__ void add(CountersOnly const &co) {
        for (uint32_t i = 0; i < totbins(); ++i) {
#ifdef __CUDA_ARCH__
          atomicAdd(off + i, co.off[i]);
#else
          auto &a = (std::atomic<Counter> &)(off[i]);
          a += co.off[i];
#endif
        }
      }

      static __host__ __device__ __forceinline__ uint32_t atomicIncrement(Counter &x) {
#ifdef __CUDA_ARCH__
        return atomicAdd(&x, 1);
#else
        auto &a = (std::atomic<Counter> &)(x);
        return a++;
#endif
      }

      static __host__ __device__ __forceinline__ uint32_t atomicDecrement(Counter &x) {
#ifdef __CUDA_ARCH__
        return atomicSub(&x, 1);
#else
        auto &a = (std::atomic<Counter> &)(x);
        return a--;
#endif
      }

      __host__ __device__ __forceinline__ void countDirect(T b) {
        assert(b < nbins());
        atomicIncrement(off[b]);
      }

      __host__ __device__ __forceinline__ void fillDirect(T b, index_type j) {
        assert(b < nbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

      __host__ __device__ __forceinline__ int32_t bulkFill(AtomicPairCounter &apc, index_type const *v, uint32_t n) {
        auto c = apc.add(n);
        if (c.m >= nbins())
          return -int32_t(c.m);
        off[c.m] = c.n;
        for (uint32_t j = 0; j < n; ++j)
          bins[c.n + j] = v[j];
        return c.m;
      }

      __host__ __device__ __forceinline__ void bulkFinalize(AtomicPairCounter const &apc) {
        off[apc.get().m] = apc.get().n;
      }

      __host__ __device__ __forceinline__ void bulkFinalizeFill(AtomicPairCounter const &apc) {
        auto m = apc.get().m;
        auto n = apc.get().n;
        if (m >= nbins()) {  // overflow!
          off[nbins()] = uint32_t(off[nbins() - 1]);
          return;
        }
        auto first = m + blockDim.x * blockIdx.x + threadIdx.x;
        for (auto i = first; i < totbins(); i += gridDim.x * blockDim.x) {
          off[i] = n;
        }
      }

      __host__ __device__ __forceinline__ void count(T t) {
        uint32_t b = bin(t);
        assert(b < nbins());
        atomicIncrement(off[b]);
      }

      __host__ __device__ __forceinline__ void fill(T t, index_type j) {
        uint32_t b = bin(t);
        assert(b < nbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

      __host__ __device__ __forceinline__ void count(T t, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        atomicIncrement(off[b]);
      }

      __host__ __device__ __forceinline__ void fill(T t, index_type j, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        bins[w - 1] = j;
      }

      __host__ __device__ __forceinline__ void finalize(Counter *ws = nullptr) {
        assert(off[totbins() - 1] == 0);
        blockPrefixScan(off, totbins(), ws);
        assert(off[totbins() - 1] == off[totbins() - 2]);
      }

      template <typename COUNT, typename FILL>
      __device__ __forceinline__ void countAndFill(COUNT count, FILL fill) {
        auto voidTail = []() {};
        tasks[0].doit(count, voidTail);
#ifdef __CUDACC__
        multiTaskPrefixScan(off, off, totbins(), tasks+1, psws);
#else
        finalize();
#endif
/*       
       auto verify = [&](int iWork) {
//        if (iWork + threadIdx.x == 0) printf("last bins %d: %d %d %d\n",totbins(),off[totbins() - 1], off[totbins() - 2], off[totbins() - 3]);
        assert(off[totbins() - 1] == off[totbins() - 2]);
        auto first = blockDim.x * iWork + threadIdx.x;
        for (auto i = first; i < totbins()-1; i += gridDim.x * blockDim.x) {
          if (off[i+1] < off[i]) printf("error %d: %d %d %d %d\n",i,off[i-1],off[i],off[i+1],off[i+2]);
//          assert(off[i+1] >=off[i]);
        }
       };
       tasks[4].doit(verify, voidTail);
*/       

        tasks[3].doit(fill, voidTail);
        //fill(blockIdx.x);
      }

      constexpr auto size() const { return uint32_t(off[totbins() - 1]); }
      constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

      constexpr index_type const *begin() const { return bins; }
      constexpr index_type const *end() const { return begin() + size(); }

      constexpr index_type const *begin(uint32_t b) const { return bins + off[b]; }
      constexpr index_type const *end(uint32_t b) const { return bins + off[b + 1]; }

      Counter off[totbins()];
      CUDATask tasks[5];  // to run count-prefixscan-fill
      index_type bins[capacity()];
      uint32_t psws[nblocks()];
    };

    template <typename I,        // type stored in the container (usually an index in a vector of the input values)
              uint32_t MAXONES,  // max number of "ones"
              uint32_t MAXMANYS  // max number of "manys"
              >
    using OneToManyAssoc = HistoContainer<uint32_t, MAXONES, MAXMANYS, sizeof(uint32_t) * 8, I, 1>;

  }  // namespace cuda
}  // namespace cms

// #undef GPU_DEBUG

#endif  // HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
