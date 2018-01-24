/**
 * Derived from the nVIDIA CUDA 8.0 samples by
 *
 *   Eyal Rozenberg <E.Rozenberg@cwi.nl>
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 *
 *  Modified by VinInn for testing math funcs
 */

/* to run test
foreach f ( $CMSSW_BASE/test/$SCRAM_ARCH/DFM_Vector* )
echo $f; $f
end
*/

#include "cuda/api_wrappers.h"

#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <chrono>
#include<random>

#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"


std::mt19937 eng;
std::mt19937 eng2;
std::uniform_real_distribution<float> rgen(0.,1.);


#include<DataFormats/Math/interface/approx_exp.h>
__host__ __device__
inline float myExp(float x) {
  return  unsafe_expf<6>(x);
}

#ifdef __NVCC__
#define inline __host__ __device__ inline
#include<vdt/sin.h>
#undef inline
#else
#include<vdt/sin.h>
#endif

__host__ __device__
inline float mySin(float x) {
  return vdt::fast_sinf(x);
}


#include<DataFormats/Math/interface/approx_log.h>
__host__ __device__ 
inline float myLog(float x) {
  return  unsafe_logf<6>(x);
}


__host__ __device__
inline float testFunc(float x, float y) { return 
#ifdef USEEXP
  myExp(x)
#elif defined(USESIN)
  mySin(x)
#else
  myLog(x)
#endif
#ifdef ADDY
+ y
#endif
;}


__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements) { C[i] = testFunc(A[i],B[i]); }
}


void vectorAddH(const float *A, const float *B, float *C, int numElements)
{
   tbb::parallel_for(
    tbb::blocked_range<size_t>(0,numElements),
    [&](const tbb::blocked_range<size_t>& r) {
      for (auto i=r.begin();i<r.end();++i)         
           { C[i] = testFunc(A[i],B[i]); }
    }
   );
}


int main(void)
{
  
    std::cerr << "default num of thread " << tbb::task_scheduler_init::default_num_threads() << std::endl;

  //tbb::task_scheduler_init init;  // Automatic number of threads
   tbb::task_scheduler_init init(tbb::task_scheduler_init::default_num_threads());  // Explicit number of threads


  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;
	if (cuda::device::count() == 0) {
		std::cerr << "No CUDA devices on this system" << "\n";
		exit(EXIT_FAILURE);
	}

	int numElements = 100000;
	size_t size = numElements * sizeof(float);
	std::cout << "[Vector evaluation of " << numElements << " elements]\n";

	// If we could rely on C++14, we would  use std::make_unique
	auto h_A = std::make_unique<float[]>(numElements);
	auto h_B = std::make_unique<float[]>(numElements);
	auto h_C = std::make_unique<float[]>(numElements);
        auto h_C2 = std::make_unique<float[]>(numElements);

	std::generate(h_A.get(), h_A.get() + numElements, [&](){return rgen(eng);});
	std::generate(h_B.get(), h_B.get() + numElements, [&](){return rgen(eng);});

        start = std::chrono::high_resolution_clock::now();
	auto current_device = cuda::device::current::get();
	auto d_A = cuda::memory::device::make_unique<float[]>(current_device, numElements);
	auto d_B = cuda::memory::device::make_unique<float[]>(current_device, numElements);
	auto d_C = cuda::memory::device::make_unique<float[]>(current_device, numElements);

	cuda::memory::copy(d_A.get(), h_A.get(), size);
	cuda::memory::copy(d_B.get(), h_B.get(), size);
        delta = (std::chrono::high_resolution_clock::now()-start);
        std::cout <<"cuda alloc+copy took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
              << " ms" << std::endl;


	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	std::cout
		<< "CUDA kernel launch with " << blocksPerGrid
		<< " blocks of " << threadsPerBlock << " threads\n";

        start = std::chrono::high_resolution_clock::now();
	for (int j=0; j<1000;++j) cuda::launch(
		vectorAdd,
		{ blocksPerGrid, threadsPerBlock },
		d_A.get(), d_B.get(), d_C.get(), numElements
	);
        delta = std::chrono::high_resolution_clock::now()-start;
        std::cout <<"cuda computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
              << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        for (int j=0; j<1000;++j) cuda::launch(
                vectorAdd,
                { blocksPerGrid, threadsPerBlock },
                d_A.get(), d_B.get(), d_C.get(), numElements
        );
        delta = std::chrono::high_resolution_clock::now()-start;
        std::cout <<"cuda computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
              << " ms" << std::endl;


	cuda::memory::copy(h_C.get(), d_C.get(), size);


        start = std::chrono::high_resolution_clock::now();
        for (int j=0; j<1000;++j) vectorAddH(h_A.get(),h_B.get(),h_C2.get(),numElements);        
        delta = std::chrono::high_resolution_clock::now()-start;
        std::cout <<"host computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
              << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        for (int j=0; j<1000;++j) {
        vectorAddH(h_A.get(),h_B.get(),h_C2.get(),numElements);
        }
        delta = std::chrono::high_resolution_clock::now()-start;
        std::cout <<"host computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
              << " ms" << std::endl;


	// Verify that the result vector is correct
        double ave = 0;
        int maxDiff = 0;
        long long ndiff=0;
        double fave = 0;
        float fmaxDiff = 0;
	for (int i = 0; i < numElements; ++i) {
                        approx_math::binary32 g,c;
                        g.f = testFunc(h_A[i],h_B[i]);
                        c.f = h_C[i];
                        auto diff = std::abs(g.i32-c.i32);
                        maxDiff = std::max(diff,maxDiff);
                        ave += diff;
                        if (diff!=0) ++ndiff;
                        auto fdiff = std::abs(g.f-c.f);
                        fave += fdiff;
                        fmaxDiff = std::max(fdiff,fmaxDiff);
//           if (diff>7)
//           std::cerr << "Large diff at element " << i << ' ' << diff << ' ' << std::hexfloat 
//                                  << g.f << "!=" << c.f << "\n";
	}
        std::cout << "ndiff ave, max " << ndiff << ' ' << ave/numElements << ' ' << maxDiff << std::endl;
        std::cout << "float ave, max " << fave/numElements << ' ' << fmaxDiff << std::endl;
        if (ndiff) exit(EXIT_FAILURE);
	std::cout << "Test PASSED\n";
	std::cout << "SUCCESS\n";
	return 0;
}

