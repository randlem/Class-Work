 cc = gcc
flags = -funroll-loops -O3
out = particle.out
source = mpi_particle.c png_writer.c main.c
libs = -lm -lz -lpng
clean_rule = rm -rf

main:
	$(cc) $(flags) $(source) -o $(out) $(libs)

clean:
	$(clean_rule) $(out)