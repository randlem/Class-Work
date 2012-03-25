#ifndef __LLSGPU_SOLVER_KERNEL_CU__
#define __LLSGPU_SOLVER_KERNEL_CU__

#define THETA_ACCUM_CNT 180
#define DATA_POINTS 6
#define F 100

#include "util.h"
#include "cudautil.h"

typedef struct {
	int x;
	int y;
} edge_pixel_t;

typedef struct {
	int x;
	int y;
	int a;
	int b;
	int theta;
} ellipse_t;

           cuda_dim        Solver_dim        = { 64, 150 };
		   edge_pixel_t*   h_edge_pixels     = NULL;
		   uint            h_edge_pixels_cnt = 0;
__device__ edge_pixel_t*   d_edge_pixels     = NULL;
__device__ uint*           d_center_accum    = NULL;
__device__ uint*           d_axes_accum      = NULL;
__device__ uint*           d_theta_accum     = NULL;

__host__ void setup_Solver() {
	cudaError_t err;

	if (h_edge_pixels == NULL) {
		debug("process_input() should be run before setup_Solver()");
		return;
	}

	err = cudaMalloc((void **)&d_edge_pixels, sizeof(edge_pixel_t) * h_edge_pixels_cnt);
	if (err != cudaSuccess)
		debug("%s",cudaGetErrorString(err));
	err = cudaMemcpy(d_edge_pixels, h_edge_pixels, sizeof(edge_pixel_t) * h_edge_pixels_cnt, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		debug("%s",cudaGetErrorString(err));
}

__host__ void setup_Solver_accum(uint width, uint height) {
	int size = width * height;
	cudaError_t err;

	// setup the 2d center accumulator
	err = cudaMalloc((void **)&d_center_accum, sizeof(uint) * size);
	if (err != cudaSuccess)
		debug("center accum malloc: %s",cudaGetErrorString(err));
	err = cudaMemset(d_center_accum, 0, sizeof(uint) * size);
	if (err != cudaSuccess)
		debug("center accum memset: %s",cudaGetErrorString(err));

	// setup the 2d axes accumulator
	err = cudaMalloc((void **)&d_axes_accum, sizeof(uint) * size);
	if (err != cudaSuccess)
		debug("axes accum malloc: %s",cudaGetErrorString(err));
	err = cudaMemset(d_axes_accum, 0, sizeof(uint) * size);
	if (err != cudaSuccess)
		debug("axes accum memset: %s",cudaGetErrorString(err));

	// setup the 1d theta accumulator
	err = cudaMalloc((void **)&d_theta_accum, sizeof(uint) * THETA_ACCUM_CNT);
	if (err != cudaSuccess)
		debug("theta accum malloc: %s",cudaGetErrorString(err));
	err = cudaMemset(d_theta_accum, 0, sizeof(uint) * THETA_ACCUM_CNT);
	if (err != cudaSuccess)
		debug("theta accum memset: %s",cudaGetErrorString(err));
}

__host__ void teardown_Solver() {
	cudaError_t err;

	if (d_edge_pixels != NULL) {
		err = cudaFree(d_edge_pixels);
		if (err != cudaSuccess)
			debug("d_edge_pixel free: %s",cudaGetErrorString(err));
		d_edge_pixels = NULL;
	}

	if (h_edge_pixels != NULL) {
		delete [] h_edge_pixels;
		h_edge_pixels = NULL;
	}
}

__host__ void teardown_Solver_accum() {
	cudaError_t err;

	if (d_theta_accum != NULL) {
		err = cudaFree(d_theta_accum);
		if (err != cudaSuccess)
			debug("theta accum free: %s",cudaGetErrorString(err));
		d_theta_accum = NULL;
	}

	if (d_axes_accum != NULL) {
		err = cudaFree(d_axes_accum);
		if (err != cudaSuccess)
			debug("axes accum free: %s",cudaGetErrorString(err));
		d_axes_accum = NULL;
	}

	if (d_center_accum != NULL) {
		err = cudaFree(d_center_accum);
		if (err != cudaSuccess)
			debug("center accum free: %s",cudaGetErrorString(err));
		d_center_accum = NULL;
	}
}

// implemented from psudocode at: http://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode
// gaussian elimination with partial pivoting
// results is ordered x1,x2,x3...xn
__device__ void gauss(double (*A)[DATA_POINTS], uint rows, uint cols, double* results) {
	int i,j,k,u,maxi;
	double var, t;

	i = 0;
	j = 0;
	while (i < rows && j < cols) {
		// find the row with the maximum value
		maxi = i;
		for (k=i+1; k < rows; k++) {
			if (fabs(A[k][j]) > fabs(A[maxi][j]))
				maxi = k;
		}

		if (A[maxi][j] != 0) {
			// swap rows
			if (i != maxi) {
				for (u=0; u < cols; u++) {
					t = A[i][u];
					A[i][u] = A[maxi][u];
					A[maxi][u] = t;
				}
			}

			// reduce pivot element to 1
			var = A[i][j];
			for (k=0; k < cols; k++)
				A[i][k] /= var;

			// remove the pivot element from all subsequent rows
			for (u=i+1; u < rows; u++) {
				var = A[u][j];
				for (k=j; k < cols; k++)
					A[u][k] -= A[i][k] * var;
			}

			i++;
		}

		j++;
	}

	// retrieve the results
	for (i=rows-1; i >= 0; i--) {
		var = A[i][cols-1];
		for (j=cols-2; j > i; j--) {
			var -= A[i][j] * results[j];
		}
		results[i] = var;
	}
}

__device__ void accumulator_3(ellipse_t* ellipse, uint* d_center_accum, uint* d_axes_accum, uint* d_theta_accum, uint width, uint height) {
	uint *center_ptr, *axes_ptr, *theta_ptr, offset;

	center_ptr = axes_ptr = theta_ptr = NULL;

	offset = ellipse->y * width + ellipse->x;
	if (offset < width * height
			&& ellipse->x >= 0 && ellipse->x < width
			&& ellipse->y >= 0 && ellipse->y < height)
		center_ptr = &d_center_accum[offset];

	offset = ellipse->b * width + ellipse->a;
	if (offset < width * height
			&& ellipse->a >= 0 && ellipse->a < width
			&& ellipse->b >= 0 && ellipse->b < height)
		axes_ptr   = &d_axes_accum[offset];

	ellipse->theta += THETA_ACCUM_CNT / 2;
	if (ellipse->theta >= 0 && ellipse->theta < THETA_ACCUM_CNT)
		theta_ptr = &d_theta_accum[ellipse->theta];

	if (center_ptr !=  NULL && axes_ptr != NULL && theta_ptr != NULL) {
		//printf("%d %d: %x %x %x\t%dx%d %dx%d %d\n", blockIdx.x,threadIdx.x, center_ptr, axes_ptr, theta_ptr,ellipse->x,ellipse->y,ellipse->a,ellipse->b,ellipse->theta-90);
		atomicAdd(center_ptr,1);
		atomicAdd(axes_ptr,1);
		atomicAdd(theta_ptr,1);
	}
}

__global__ void Solver_kernel(
	uint cnt,
	edge_pixel_t* d_edge_pixels,
	uint d_edge_pixels_cnt,
	uint* d_rnds,
	uint* d_center_accum,
	uint* d_axes_accum,
	uint* d_theta_accum,
	uint width,
	uint height
) {
	int next_rnd, x2, y2, xy, rnd, i, j, m;
	double X[DATA_POINTS][5], Xt[5][DATA_POINTS], aug[5][DATA_POINTS], results[5];
	double A, B, C, D, E, J, delta, t, r1, r2, slope1, slope2;
	edge_pixel_t* pxl;
	ellipse_t ellipse;

	// figure out how many pieces of evidence to collect
	next_rnd = (blockIdx.x * blockDim.x + threadIdx.x) * cnt * 6;

	// compute the requried evidence
	for(; cnt > 0; cnt--) {
		// generate X and Xt from random edge points
		for(i=0; i < DATA_POINTS; i++) {
			rnd = (d_rnds[next_rnd++] / (float)0xFFFFFFFF) * (d_edge_pixels_cnt);
			pxl = &d_edge_pixels[rnd];
			x2  = pxl->x * pxl->x;
			y2  = pxl->y * pxl->y;
			xy  = pxl->x * pxl->y;

			Xt[0][i] = X[i][0] = x2;
			Xt[1][i] = X[i][1] = y2;
			Xt[2][i] = X[i][2] = xy;
			Xt[3][i] = X[i][3] = pxl->x;
			Xt[4][i] = X[i][4] = pxl->y;
		}

		// generate the augmented matrix from X, Xt, and Y
		for (i=0; i < 5; i++) {
			for (j=0; j < 5; j++) {
				aug[i][j] = 0.0;
				for (m=0; m < 6; m++)
					aug[i][j] += Xt[i][m] * X[m][j];
			}
		}
		for (i=0; i < 5; i++) {
			aug[i][5] = 0.0;
			for (m=0; m < 6; m++)
				aug[i][5] += Xt[i][m] * F;
		}

		// solve the setup general quadratic
		gauss(aug,5,6,results);
		A = results[0]; B = results[1]; C = results[2]; D = results[3]; E = results[4];

		// calc j to determine if we have an ellipse
		J = (A * B) - ((C * C) / 4.0);

		// determine if we have a circle
		if (J > 0.0 || fabs(J) <= FP_PRE) {
			// recover the parameters
			delta = (A * B * -F)     +
					(C * E * D)  / 8 +
					(D * C * E)  / 8 -
					(D * D * B)  / 4 -
					(A * E * E)  / 4 -
					(C * C * -F) / 4;
			t     = sqrt((B - A) * (B - A) + C * C);
			r1    = (A + B + t) / 2.0;
			r2    = (A + B - t) / 2.0;

			t             = (C * C - 4.0 * A * B);
			ellipse.x     = (2.0 * B * D - C * E) / t;
			ellipse.y     = (2.0 * A * E - C * D) / t;
			ellipse.a     = (int)floor(sqrt(fabs(delta) / fabs(J * r2)));
			ellipse.b     = (int)floor(sqrt(fabs(delta) / fabs(J * r1)));
			t             = (B - A) / C;
			slope1        = sqrt((t * t) + 1.0) + t;
			slope2        = -sqrt((t * t) + 1.0) + t;
			ellipse.theta = atan(slope2) / M_PI * 180.0;

			if (ellipse.b > ellipse.a) {
				t = ellipse.a;
				ellipse.a = ellipse.b;
				ellipse.b = t;
				ellipse.theta = (int)(atan(slope1) / M_PI * 180.0 + 0.5);

				if (C < 0)
					ellipse.theta += 90;

			}

			accumulator_3(&ellipse,d_center_accum,d_axes_accum,d_theta_accum,width,height);
		}
	}

	// make sure everybody is done before we go
	__syncthreads();
}

#endif
