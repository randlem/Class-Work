mm1: mm1.o lcgrand.o
	gcc -o mm1 mm1.o lcgrand.o -lm
	
mm1.o: mm1.c
	gcc -c -o mm1.o mm1.c
	
lcgrand.o: lcgrand.c
	gcc -c -o lcgrand.o lcgrand.c
	
.PHONY: clean run
clean:
	@rm -rf mm1.o lcgrand.o
	
run:
	@mm1 < mm1.in
	@cat mm1.out
	@echo
