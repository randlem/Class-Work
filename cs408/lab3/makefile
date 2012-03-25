INCLUDE_PATH = -Isrc/simlib -Isrc

FLAGS = -g -Wno-deprecated

LIBS = -lm -lpthread

LIB_OBJ = SimPlus.o SampST.o Event.o EventPool.o Entity.o EntityQueue.o         \
	   ServerEntity.o EventHeap.o LocalRNG.o NormalDist.o ExponentialDist.o     \
	   UniformDist.o HTTPSocket.o NetRNG.o RNGFactory.o RawDist.o PoissonDist.o \
	   BinomialDist.o FileRNG.o LogFile.o BernoulliDist.o WeibullDist.o         \
	   LogLogisticDist.o TriangleDist.o

vpath %.cpp src/simlib:src
vpath %.h src:src/simlib:src/processor

# compile directive for anything in simlib
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) $(INCLUDE_PATH) -c $< -o $@

sim: main.o $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) $(FLAGS) -o sim $^ $(LIBS)

main.o: main.cpp processor.h dummy.h fcfs.h hrrn.h rr.h sjn.h srtn.h
	$(CXX) $(CXXFLAGS) $(FLAGS) $(INCLUDE_PATH) -c $< -o $@

.PHONY: clean
clean:
	@rm -f *.o sim
