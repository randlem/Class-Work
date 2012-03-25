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

#include "maxredux_kernel.cu"

int main(int argc, char* argv[]) {
	uint *h_in, *d_in;
	int *h_out, *d_out;
	int size = 65536, i, maxi=0;
	hr_timer_t timer;

	srand(137);

	h_in  = new uint[size];
	h_out = new int[1];

	cudaMalloc((void **)&d_in,sizeof(int) * size);
	cudaMalloc((void **)&d_out,sizeof(int) * 1);

	for(i=0; i < size; i++) {
		h_in[i] = rand();

		if (h_in[i] > h_in[maxi])
			maxi = i;
	}
	printf("max is h_in[%d] = %d\n",maxi,h_in[maxi]);

	cudaMemcpy(d_in,h_in,sizeof(int) * size,cudaMemcpyHostToDevice);
	timer_start(&timer);
	MaxRedux_kernel<512><<<1,512,sizeof(int) * 512>>>(d_in, size, d_out);
	timer_end(&timer);
	cudaMemcpy(h_out,d_out,sizeof(int) * 1,cudaMemcpyDeviceToHost);
	printf("recovered max is h_in[%d] = %d\n", h_out[0], h_in[h_out[0]]);
	print_time(&timer.elapsed, "timer ellapsed = ");

	cudaFree(d_in);
	cudaFree(d_out);
	delete [] h_out;
	delete [] h_in;
}
