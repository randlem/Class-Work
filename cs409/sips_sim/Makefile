 cc = gcc
flags = -g -Wall
out = sips_sim
source = main.c emu.c
libs =
clean_rule = rm -rf

main:
	$(cc) $(flags) $(source) -o $(out) $(libs)

clean:
	$(clean_rule) $(out)