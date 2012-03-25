cc = ecc
flags =
out = particle.out
source = particle_problem_single.c
libs = -lm
clean_rule = rm -rf
imgmkr_source = imgmkr_velocity.c
imgmkr_out = imgmkr.out
imgmkr_libs = $(libs) -lpng -lz

main:
	$(cc) $(flags) $(source) -o $(out) $(libs)

clean:
	$(clean_rule) $(out)

imgmkr:
	$(cc) $(flags) $(imgmkr_source) -o $(imgmkr_out) $(imgmkr_libs)
