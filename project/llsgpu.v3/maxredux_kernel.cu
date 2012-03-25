#ifndef __LLSGPU_MAXREDUX_KERNEL_CU__
#define __LLSGPU_MAXREDUX_KERNEL_CU__

#include "util.h"
#include "cudautil.h"

// from http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/reduction/doc/reduction.pdf
template <uint blockSize>
__global__ void MaxRedux_kernel(uint *d_in, int size, int *d_out) {
	extern __shared__ int sdata[];

	uint tid      = threadIdx.x;
	uint i        = blockIdx.x * (blockSize * 2) + tid;
	uint gridSize = blockSize * 2 * gridDim.x;

	sdata[tid] = i; i+=gridSize;
	while (i < size) {
		if (d_in[sdata[tid]] < d_in[i])
			sdata[tid] = i;
		i+=gridSize;
	}
	__syncthreads();

	if (blockSize >= 512) {
		if (tid < 256) {
			if (d_in[sdata[tid]] < d_in[sdata[tid + 256]])
				sdata[tid] = sdata[tid + 256];
		}
	}
	__syncthreads();

	if (blockSize >= 256) {
		if (tid < 128) {
			if (d_in[sdata[tid]] < d_in[sdata[tid + 128]])
				sdata[tid] = sdata[tid + 128];
		}
	}
	__syncthreads();

	if (blockSize >= 128) {
		if (tid < 64) {
			if (d_in[sdata[tid]] < d_in[sdata[tid + 64]])
				sdata[tid] = sdata[tid + 64];
		}
	}
	__syncthreads();

	if (tid < 32) {
		if (blockSize >= 64) {
			if (d_in[sdata[tid]] < d_in[sdata[tid + 32]])
				sdata[tid] = sdata[tid + 32];
		}
		if (blockSize >= 32) {
			if (d_in[sdata[tid]] < d_in[sdata[tid + 16]])
				sdata[tid] = sdata[tid + 16];
		}
		if (blockSize >= 16) {
			if (d_in[sdata[tid]] < d_in[sdata[tid + 8]])
				sdata[tid] = sdata[tid + 8];
		}
		if (blockSize >= 8) {
			if (d_in[sdata[tid]] < d_in[sdata[tid + 4]])
				sdata[tid] = sdata[tid + 4];
		}
		if (blockSize >= 4) {
			if (d_in[sdata[tid]] < d_in[sdata[tid + 2]])
				sdata[tid] = sdata[tid + 2];
		}
		if (blockSize >= 2) {
			if (d_in[sdata[tid]] < d_in[sdata[tid + 1]])
				sdata[tid] = sdata[tid + 1];
		}
	}

	if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

#endif
