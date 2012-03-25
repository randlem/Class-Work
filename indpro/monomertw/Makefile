FLAGS=-Wno-deprecated

LIBS=-lm -lpng -lmpich

OBJS=lattice.o mpiwrapper.o

HDRS=lattice.h exception.h latprim.h latconst.h randgen.h rewindlist.h mpiwrapper.h

monomertw: $(OBJS) main.o
	${CXX} $(FLAGS) -o monomertw $(OBJS) main.o $(LIBS)

testglobal: mpiwrapper.o testglobal.o
	${CXX} $(FLAGS) -o testglobal mpiwrapper.o testglobal.o $(LIBS)

main.o: main.cpp $(HDRS)
	${CXX} $(FLAGS) -c -o main.o main.cpp

lattice.o: lattice.cpp $(HDRS)
	${CXX} $(FLAGS) -c -o lattice.o lattice.cpp

mpiwrapper.o: mpiwrapper.cpp $(HDRS)
	${CXX} $(FLAGS) -c -o mpiwrapper.o mpiwrapper.cpp

testglobal.o: testglobal.cpp $(HDRS)
	${CXX} $(FLAGS) -c -o testglobal.o testglobal.cpp	

.PHONY: clean mpiclean count fileclean

clean:
	@rm -rf $(OBJS) main.o testglobal.o monomertw testglobal

mpiclean:
	@rm -rf monomertw.o* testglobal.o*

count:
	@wc -l *.cpp *.h
	
fileclean:
	@rm -rf height-node.*.png log.*.txt
