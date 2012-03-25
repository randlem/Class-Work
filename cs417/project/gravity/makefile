cc = gcc
flags = -Wall -funroll-loops -O3
out = particle.out
source = particle_problem_main.c
libs = -lm -lpng -lz
clean_rule = rm -rf

main:
	$(cc) $(flags) $(source) -o $(out) $(libs)

clean:
	$(clean_rule) $(out)