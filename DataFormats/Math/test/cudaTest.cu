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
 */

#include "cuda/api_wrappers.h"

#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>

#include<DataFormats/Math/interface/approx_log.h>
__host__ __device__ 
inline float myLog(float x) {
  return  unsafe_logf<6>(x);
}

__host__ __device__
inline float testFunc(float x, float y) { return myLog(x)
#ifdef ADDY
+ y
#endif
;}


__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements) { C[i] = testFunc(A[i],B[i]); }
}

int main(void)
{
	if (cuda::device::count() == 0) {
		std::cerr << "No CUDA devices on this system" << "\n";
		exit(EXIT_FAILURE);
	}

	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	std::cout << "[Vector addition of " << numElements << " elements]\n";

	// If we could rely on C++14, we would  use std::make_unique
	auto h_A = std::make_unique<float[]>(numElements);
	auto h_B = std::make_unique<float[]>(numElements);
	auto h_C = std::make_unique<float[]>(numElements);

	auto generator = []() { return rand() / (float) RAND_MAX; };
	std::generate(h_A.get(), h_A.get() + numElements, generator);
	std::generate(h_B.get(), h_B.get() + numElements, generator);

	auto current_device = cuda::device::current::get();
	auto d_A = cuda::memory::device::make_unique<float[]>(current_device, numElements);
	auto d_B = cuda::memory::device::make_unique<float[]>(current_device, numElements);
	auto d_C = cuda::memory::device::make_unique<float[]>(current_device, numElements);

	cuda::memory::copy(d_A.get(), h_A.get(), size);
	cuda::memory::copy(d_B.get(), h_B.get(), size);

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	std::cout
		<< "CUDA kernel launch with " << blocksPerGrid
		<< " blocks of " << threadsPerBlock << " threads\n";

	cuda::launch(
		vectorAdd,
		{ blocksPerGrid, threadsPerBlock },
		d_A.get(), d_B.get(), d_C.get(), numElements
	);

	cuda::memory::copy(h_C.get(), d_C.get(), size);

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

