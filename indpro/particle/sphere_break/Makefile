 cc = gcc
flags = -funroll-loops -O3
out = sphere.out
source = main1.c png_writer.c
libs = -lm -lz -lpng
clean_rule = rm -rf

main:
	$(cc) $(flags) $(source) -o $(out) $(libs)

clean:
	$(clean_rule) $(out)