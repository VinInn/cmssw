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

#include<cassert>

#ifdef __NVCC__
#define inline __host__ __device__ inline
#include<DataFormats/Math/interface/approx_exp.h>
#include<DataFormats/Math/interface/approx_log.h>
#include<DataFormats/Math/interface/approx_atan2.h>
#undef inline
#else
#include<DataFormats/Math/interface/approx_exp.h>
#include<DataFormats/Math/interface/approx_log.h>
#include<DataFormats/Math/interface/approx_atan2.h>
#endif

  constexpr float xmin=-100.001;  // avoid 0
  constexpr float incr = 0.04;
  constexpr int Nsteps = 2.*std::abs(xmin)/incr;


// #define DEGREE 9
template<int DEGREE>
__global__ void diffAtan(int * diffs) {


  auto mdiff = &diffs[0];
  auto idiff = &diffs[1];
  auto sdiff = &diffs[2];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  auto x = xmin +incr*i;
  auto y = xmin	+incr*j;

  auto approx = unsafe_atan2f<DEGREE>(y,x);
  auto iapprox = unsafe_atan2i<DEGREE>(y,x);
  auto sapprox = unsafe_atan2s<DEGREE>(y,x);
  auto std = std::atan2(y,x);
  auto fd = std::abs(std-approx);
  atomicMax(mdiff, int(fd*1.e7) );
  atomicMax(idiff, std::abs(phi2int(std)-iapprox));
  short dd = std::abs(phi2short(std)-sapprox);
  atomicMax(sdiff,int(dd));

}

template<int DEGREE>
void go() {
  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;

	if (cuda::device::count() == 0) {
		std::cerr << "No CUDA devices on this system" << "\n";
		exit(EXIT_FAILURE);
	}

        auto current_device = cuda::device::current::get(); 
        // atan2
        delta -= (std::chrono::high_resolution_clock::now()-start);
 
        auto diff_d = cuda::memory::device::make_unique<int[]>(current_device,3);
 
        int diffs[3];
        cuda::memory::device::zero(diff_d.get(),3*4);

        // Launch the diff CUDA Kernel
        dim3 threadsPerBlock(32,32,1);
        dim3 blocksPerGrid ( (Nsteps + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (Nsteps + threadsPerBlock.y - 1) / threadsPerBlock.y,
                            1
                               );
        std::cout
                << "CUDA kernel 'diff' launch with " << blocksPerGrid.x
                << " blocks of " << threadsPerBlock.y << " threads\n";

        cuda::launch(
                diffAtan<DEGREE>,
                { blocksPerGrid, threadsPerBlock },
                diff_d.get() );

        cuda::memory::copy(diffs, diff_d.get(), 3*4);
        delta += (std::chrono::high_resolution_clock::now()-start);
 
        float mdiff = diffs[0]*1.e-7;
        int idiff = diffs[1];
        int sdiff = diffs[2];

        std::cout << "for degree " << DEGREE << " max diff is " << mdiff 
                             << ' ' << idiff << ' ' << int2phi(idiff) 
                             << ' ' << sdiff << ' ' << short2phi(sdiff) <<  std::endl;


        std::cout <<"cuda computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
              << " ms" << std::endl;

	return;
}


int main() {

  go<3>();
  go<5>();
  go<7>();
  go<9>();


  return 0;
}
