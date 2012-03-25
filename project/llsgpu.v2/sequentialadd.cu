#define DEBUG 1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <math_functions.h>

#define STD_BLOCK_SIZE 10
#define COLS 10

__global__ void sequential_add(int* A, int cols) {
	for(int i=0; i < 20; i++)
		atomicAdd(&A[threadIdx.x],1);
}

void debug(const char * s, ...) {
	if (DEBUG > 0) {
		va_list args;
		va_start(args, s);
		vfprintf(stdout, s, args);
		fprintf(stdout, "\n");
		va_end(args);
	}
}

int main(int argc, char* argv[]) {
	int *A_d;
	int *A_h;
	int i, cols;
	int block_size = STD_BLOCK_SIZE, grid_size;
	cudaError_t err;

	// allocate the host arrays
	debug("allocate host memory");
	A_h = (int *)malloc(sizeof(int) * COLS);

	// allocate the device arrays
	debug("allocate device memory");
	err = cudaMalloc((void **)&A_d,sizeof(int) * COLS);
	if (err != cudaSuccess) {
		switch(err) {
			case cudaErrorMemoryAllocation:
				debug("Memory Allocation");
				break;
		}
		return -1;
	}

	// blank A
	memset(A_h,0,sizeof(int) * COLS);

	// create the block and grid data structs
	dim3 dimBlock(block_size);
	grid_size = 10;
	dim3 dimGrid(grid_size);
	printf("Created grid size %d in block size %d\n",grid_size,block_size);

	// copy A and B to the device
	debug("copy host to device memory");
	err = cudaMemcpy(A_d,A_h,sizeof(int) * COLS,cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		switch(err) {
			case cudaErrorInvalidValue:
				debug("Invalid Value");
				break;
			case cudaErrorInvalidDevicePointer:
				debug("Invalid Device Pointer");
				break;
			case cudaErrorInvalidMemcpyDirection:
				debug("Invalid Memcpy Direction");
				break;
		}
		return -1;
	}

	// run the kernel
	debug("run kernel");
	sequential_add<<<dimGrid,dimBlock>>>(A_d, COLS);
	cudaThreadSynchronize();

	// copy the C array off of the device
	debug("copy device to host memory");
	err = cudaMemcpy(A_h,A_d,sizeof(int) * COLS, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		switch(err) {
			case cudaErrorInvalidValue:
				debug("Invalid Value");
				break;
			case cudaErrorInvalidDevicePointer:
				debug("Invalid Device Pointer");
				break;
			case cudaErrorInvalidMemcpyDirection:
				debug("Invalid Memcpy Direction");
				break;
		}
		return -1;
	}

	for(i=0; i < COLS; i++) {
		printf("%d ", A_h[i]);
	}
	printf("\n");

	// cleanup device memory
	cudaFree(A_d);

	// cleanup host memory
	free(A_h);

	return 0;
}
