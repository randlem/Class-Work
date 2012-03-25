#define DEBUG 1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>

#define ROWS 5000
#define COLS 5000

void matrix_multiply(int* A, int* B, int* C, int rows, int cols) {
	int i, j, k;

	for (i=0; i < rows; i++) {
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
	int *A_h, *B_h, *C_h;
	int i, j;
	timespec start, end, diff;

	// allocate the host arrays
	printf("Trying to allocate memory for host arrays...");
	A_h = (int *)malloc(sizeof(int) * ROWS * COLS);
	B_h = (int *)malloc(sizeof(int) * ROWS * COLS);
	C_h = (int *)malloc(sizeof(int) * ROWS * COLS);
	printf("success!\n");

	// fill A with random numbers
	for (i=0; i < ROWS; i++)
		for (j=0; j < COLS; j++)
			A_h[i * COLS + j] = random_int(0,100);

	// fill B with random numbers
	for (i=0; i < ROWS; i++)
		for (j=0; j < COLS; j++)
			B_h[i * COLS + j] = random_int(0,100);

	// run the kernel
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
	matrix_multiply(A_h, B_h, C_h, ROWS, COLS);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
	time_diff(&start,&end,&diff);
	print_time(&diff,"matrix_multiply");

	// cleanup host memory
	free(A_h);
	free(B_h);
	free(C_h);

	return 0;
}
