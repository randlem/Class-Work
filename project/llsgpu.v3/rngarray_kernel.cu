#ifndef __LLSGPU_RNGARRAY_KERNEL_CU__
#define __LLSGPU_RNGARRAY_KERNEL_CU__

#include "util.h"
#include "cudautil.h"
#include "mt19937.h"

__device__ uint*    d_rnds;
           cuda_dim RNGArray_dim = { 1, 227 };

__host__ void setup_RNGArray(uint num) {
	cudaError_t err;

	err = cudaMalloc((void **)&d_rnds, sizeof(uint) * num);
	if (err != cudaSuccess)
		debug("d_rnds alloc: %s",cudaGetErrorString(err));
	err = cudaMemset(d_rnds, 0, sizeof(uint) * num);
	if (err != cudaSuccess)
		debug("d_rnds memset: %s",cudaGetErrorString(err));
}

__host__ void teardown_RNGArray() {
	cudaError_t err;

	if (d_rnds != NULL) {
		err = cudaFree((void *)d_rnds);
		if (err != cudaSuccess)
			debug("d_rnds free: %s",cudaGetErrorString(err));
		d_rnds = NULL;
	}
}

__global__ void RNGArray_kernel(uint* d_rnds, uint num) {
	int i, iters = num / blockDim.x, rem = num % blockDim.x;
	uint final;

	// seed the RNG
	mt19937gi(clock());

	// generate the numbers in increasing order
	for(i=0; i < iters; i++) {
		d_rnds[i * blockDim.x + threadIdx.x] = mt19937g();
	}
	__syncthreads();

	final = mt19937g();
	if (rem > 0 && threadIdx.x < rem) {
		d_rnds[iters * blockDim.x + threadIdx.x] = final;
	}
	__syncthreads();
}

#endif
