CC=g++

mm1: mm1.o lcgrand.o simulation.o
	$(CC) -o mm1 mm1.o lcgrand.o simulation.o -lm

mm1.o: mm1.cpp
	$(CC) -c -o mm1.o mm1.cpp

lcgrand.o: lcgrand.c
	$(CC) -c -o lcgrand.o lcgrand.c

simulation.o: simulation.cpp
	$(CC) -c -o simulation.o simulation.cpp

.PHONY: clean run
clean:
	@rm -rf mm1.o lcgrand.o simulation.o

run:
	@mm1 < mm1.in
	@cat mm1.out
	@echo
