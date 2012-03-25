CXX = gcc
FLAGS = -g
LIBS = -lm

HDRS = pik.h

pik-new: $(HDRS) pik.o main-new.o
	$(CXX) $(FLAGS) pik.o main-new.o -o pik-new $(LIBS)

pik-std: $(HDRS) pik.o main-std.o
	$(CXX) $(FLAGS) pik.o main-std.o -o pik-std $(LIBS)

pik-mpi: $(HDRS) pik.o main-mpi.o
	$(CXX) $(FLAGS) pik.o main-mpi.o -o pik-mpi $(LIBS)

pik.o: $(HDRS) pik.c
	${CXX} $(FLAGS) -c $*.c -o $@

main-new.o: $(HDRS) main-new.c
	${CXX} $(FLAGS) -c $*.c -o $@

main-std.o: $(HDRS) main-std.c
	${CXX} $(FLAGS) -c $*.c -o $@

main-mpi.o: $(HDRS) main-mpi.c
	${CXX} $(FLAGS) -c $*.c -o $@

.PHONY:	clean count
clean:
	@/usr/bin/rm -f pik-new pik-std main-new.o main-std.o pik.o
count:
	@/usr/bin/wc -l *.c *.h