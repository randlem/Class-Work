#define DEBUG		  0
#define FP_PRE        0.0000001
#define USAGE_MESSAGE "Usage: llsgpu <file>"
#define DATA_POINTS   6

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
#include "imgutil.h"

#include "rngarray_kernel.cu"
#include "solver_kernel.cu"
#include "maxredux_kernel.cu"

// RNGArray vars
extern cuda_dim RNGArray_dim;
uint            rnds_cnt = Solver_dim.grid_dim.x * Solver_dim.block_dim.x * 6;

// Solver vars
extern edge_pixel_t* h_edge_pixels;

// local global vars
img_t in_image;

void setup(int, char*[]);
void process_input(img_t*);
void teardown();

int main(int argc, char* argv[]) {
	hr_timer_t rng_time, solver_time, accum_process;
	int *d_results, *h_results, i;
	uint *h_theta_accum;
	ellipse_t ellipse = {0, 0, 0, 0, 0};
	cudaError_t err;

	// setup the detector
	setup(argc,argv);

	// generate all of the random numbers needed
	timer_start(&rng_time);
	RNGArray_kernel<<<RNGArray_dim.grid_dim,RNGArray_dim.block_dim>>>(d_rnds,rnds_cnt);
	err = cudaThreadSynchronize();
	if (err != cudaSuccess) {
		debug("RNGArray_kernel: %s",cudaGetErrorString(err));
		exit(-1);
	}
	timer_end(&rng_time);

	timer_start(&solver_time);
	Solver_kernel<<<Solver_dim.grid_dim,Solver_dim.block_dim>>>(1,
		d_edge_pixels,
		h_edge_pixels_cnt,
		d_rnds,
		d_center_accum,
		d_axes_accum,
		d_theta_accum,
		in_image.width,in_image.height
	);
	err = cudaThreadSynchronize();
	if (err != cudaSuccess) {
		debug("Solver_kernel: %s",cudaGetErrorString(err));
		exit(-1);
	}
	timer_end(&solver_time);

	h_theta_accum = new uint[THETA_ACCUM_CNT];
	h_results = new int[2];
	err = cudaMalloc((void **)&d_results, sizeof(int) * 2);
	if (err != cudaSuccess)
		debug("d_results malloc: %s",cudaGetErrorString(err));

	err = cudaMemcpy(h_theta_accum,
					 d_theta_accum,
	                 sizeof(uint) * THETA_ACCUM_CNT,
	                 cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		debug("d_accum->theta => h_accum->theta: %s",cudaGetErrorString(err));

	memset(h_results,0,sizeof(int) * 2);
	err = cudaMemset(d_results,0,sizeof(int) * 2);
	if (err != cudaSuccess)
		debug("d_results memset: %s",cudaGetErrorString(err));

	timer_start(&accum_process);
	MaxRedux_kernel<512><<<1, 512, sizeof(int) * 512>>>(d_center_accum, in_image.width * in_image.height, d_results);
	err = cudaThreadSynchronize();
	if (err != cudaSuccess) {
		debug("MaxRedux_kernel(1): %s",cudaGetErrorString(err));
		exit(-1);
	}
	MaxRedux_kernel<512><<<1, 512, sizeof(int) * 512>>>(d_axes_accum, in_image.width * in_image.height, d_results+1);
	err = cudaThreadSynchronize();
	if (err != cudaSuccess) {
		debug("MaxRedux_kernel(2): %s",cudaGetErrorString(err));
		exit(-1);
	}
	cudaMemcpy(h_results, d_results, sizeof(int) * 2, cudaMemcpyDeviceToHost);

	if (h_results[0] > 0) {
		ellipse.y = (int)floorf(h_results[0] / in_image.width);
		ellipse.x = h_results[0] - (int)(ellipse.y * in_image.width);
	}

	if (h_results[1] > 0) {
		ellipse.b = (int)floor(h_results[1] / in_image.width);
		ellipse.a = h_results[1] - (int)(ellipse.b * in_image.width);
	}

	// the theta array is too small to process on the GPU
	ellipse.theta = 0;
	for(i=1; i < THETA_ACCUM_CNT; i++) {
		if (h_theta_accum[ellipse.theta] < h_theta_accum[i])
			ellipse.theta = i;
	}
	ellipse.theta -= (THETA_ACCUM_CNT / 2);

	timer_end(&accum_process);

	cudaFree(d_results);
	delete [] h_results;

	printf("%f %f %f %f %d %d %d %d %d\n",
		compute_secs(&rng_time.elapsed),
		compute_secs(&solver_time.elapsed),
		compute_secs(&accum_process.elapsed),
		compute_secs(&rng_time.elapsed) + compute_secs(&solver_time.elapsed) + compute_secs(&accum_process.elapsed),
		ellipse.x, ellipse.y,
		ellipse.a, ellipse.b,
		ellipse.theta
	);

	// exit cleanly
	//teardown();
	return 0;
}

void setup(int argc, char* argv[]) {

	// check the lenght of the cmd line
	if (argc != 2)
		fprintf(stderr,"%s\n",USAGE_MESSAGE);

	// read in the input image
	image_read(argv[1],&in_image);

	// setup the RNG for our random numbers
	setup_RNGArray(rnds_cnt);

	// setup the sovler
	process_input(&in_image);
	setup_Solver();
	setup_Solver_accum(in_image.width,in_image.height);
}

void process_input(img_t* input) {
	int y, x, i;

	h_edge_pixels_cnt = 0;
	for (y=0; y < input->height; y++) {
		for(x=0; x < input->width; x++) {
			if (input->pixels[y][x].rgba != 0)
				h_edge_pixels_cnt++;
		}
	}

	h_edge_pixels = new edge_pixel_t[h_edge_pixels_cnt];
	memset(h_edge_pixels,0,sizeof(edge_pixel_t) * h_edge_pixels_cnt);

	i = 0;
	for (y=0; y < input->height; y++) {
		for(x=0; x < input->width; x++) {
			if (input->pixels[y][x].rgba != 0) {
				h_edge_pixels[i].x = x;
				h_edge_pixels[i].y = y;
				i++;
			}
		}
	}
}

void teardown() {
	teardown_Solver_accum();
	teardown_Solver();
	teardown_RNGArray();
}
