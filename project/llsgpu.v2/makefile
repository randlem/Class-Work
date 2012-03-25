llsgpu: main.cu imgutil.h util.h cudautil.h rngarray_kernel.cu solver_kernel.cu
	nvcc -arch=compute_11 -code=compute_11 -O2 main.cu -o llsgpu -lpng -lglut -lrt

debug: main.cu imgutil.h util.h cudautil.h rngarray_kernel.cu solver_kernel.cu
	nvcc -g -deviceemu -arch=compute_11 -code=compute_11 main.cu -o llsgpu -lpng -lglut -lrt

clean:
	rm llsgpu

run:
#	./llsgpu ellipse-10.png
	./llsgpu test/ellipse-single-basic.png
