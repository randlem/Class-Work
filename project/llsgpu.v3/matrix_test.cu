#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <math_functions.h>

#include "util.h"
#include "cudautil.h"

typedef struct {
	int   rows;
	int   cols;
	float* vals;
} cuda_matrix_t;

__global__ void matrix_test() {
	cuda_matrix_t A = {2, 3, {0,0,0,0,0}},
				  B = {2, 3},
				  C = {2, 3};

}

int main (int argc, char *argv[]) {

	matrix_test<<<1,10>>>();

}
