#define DEBUG 1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <math_functions.h>

#define STD_BLOCK_SIZE 512

#define ROWS 5000
#define COLS 5000

__global__ void matrix_multiply(int* A, int* B, int* C, int rows, int cols) {
	int start_row, num_rows, offset, i, j, k;

	num_rows  = (int)floorf(rows / blockDim.x);
	offset    = rows - (num_rows * blockDim.x);
	if (threadIdx.x == 0) {
		num_rows += offset;
		start_row = (blockIdx.x * rows);
	} else
		start_row = (blockIdx.x * rows) + (threadIdx.x * num_rows) + offset;

	for (i=start_row; i < start_row + num_rows; i++) {
		for(j=0; j < cols; j++) {
			C[i * cols + j] = 0;
			for(k=0; k < cols; k++) {
				C[i * cols + j] += A[i * cols + k] * B[k * cols + j];
			}
		}
	}

}

void time_diff(timespec* start, timespec* end, timespec* diff) {
	if ((end->tv_nsec - start->tv_nsec) < 0) {
		diff->tv_sec  = end->tv_sec - start->tv_sec - 1;
		diff->tv_nsec = 1000000000 + end->tv_nsec - start->tv_nsec;
	} else {
		diff->tv_sec  = end->tv_sec - start->tv_sec;
		diff->tv_nsec = end->tv_nsec - start->tv_nsec;
	}
}

int random_int(int low, int high) {
	return (rand() % high) + low;
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

void print_time(const timespec *t, const char *s) {
	double computed_time = t->tv_sec + (t->tv_nsec / 1000000000.0);
	printf("%s: %f\n", s, computed_time);
}

int main(int argc, char* argv[]) {
	int *A_d, *B_d, *C_d;
	int *A_h, *B_h, *C_h;
	int i, j;
	int block_size = STD_BLOCK_SIZE, grid_size;
	timespec start, end, diff;

	// allocate the host arrays
	printf("Trying to allocate memory for host arrays...");
	A_h = (int *)malloc(sizeof(int) * ROWS * COLS);
	B_h = (int *)malloc(sizeof(int) * ROWS * COLS);
	C_h = (int *)malloc(sizeof(int) * ROWS * COLS);
	printf("success!\n");

	// allocate the device arrays
	printf("Trying to allocate memory for device arrays...");
	cudaMalloc((void **)&A_d,sizeof(int) * ROWS * COLS);
	cudaMalloc((void **)&B_d,sizeof(int) * ROWS * COLS);
	cudaMalloc((void **)&C_d,sizeof(int) * ROWS * COLS);
	printf("success!\n");

	// create the block and grid data structs
	dim3 dimBlock(block_size);
	grid_size = (int)ceil(ROWS / (float)dimBlock.x);
	dim3 dimGrid(grid_size);
	printf("Created grid size %d in block size %d",grid_size,block_size);

	// fill A with random numbers
	for (i=0; i < ROWS; i++)
		for (j=0; j < COLS; j++)
			A_h[i * COLS + j] = random_int(0,100);

	// fill B with random numbers
	for (i=0; i < ROWS; i++)
		for (j=0; j < COLS; j++)
			B_h[i * COLS + j] = random_int(0,100);

	// copy A and B to the device
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
	cudaMemcpy(A_d,A_h,sizeof(int) * ROWS * COLS,cudaMemcpyHostToDevice);
	cudaMemcpy(B_d,B_h,sizeof(int) * ROWS * COLS,cudaMemcpyHostToDevice);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
	time_diff(&start,&end,&diff);
	print_time(&diff,"cudaMemcpy to device");

	// run the kernel
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
	matrix_multiply<<<dimGrid,dimBlock>>>(A_d, B_d, C_d, ROWS, COLS);
	cudaThreadSynchronize();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
	time_diff(&start,&end,&diff);
	print_time(&diff,"matrix_multiply");

	// copy the C array off of the device
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
	cudaMemcpy(C_h,C_d,sizeof(int) * ROWS * COLS, cudaMemcpyDeviceToHost);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
	time_diff(&start,&end,&diff);
	print_time(&diff,"cudaMemcpy from device");

	// cleanup device memory
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	// cleanup host memory
	free(A_h);
	free(B_h);
	free(C_h);

	return 0;
}
