FLAGS=-Wno-deprecated -g
LIBS=-lm -lpng -lpthread
OBJS=router.o client.o sampst.o image.o weibull.o uniform.o exponential.o rngfactory.o rawrng.o localrng.o netrng.o filerng.o httpsocket.o rawdist.o
HDRS=router.h client.h base.h address.h
SIMPLUS=./simplus

lab2: $(OBJS) main.o
	${CXX} $(FLAGS) -o lab2 $(OBJS) main.o $(LIBS)

main.o: main.cpp $(HDRS)
	${CXX} $(FLAGS) -c -o main.o main.cpp

router.o: router.cpp $(HDRS)
	${CXX} $(FLAGS) -c -o router.o router.cpp

client.o: client.cpp $(HDRS)
	${CXX} $(FLAGS) -c -o client.o client.cpp

sampst.o: $(SIMPLUS)/SampST.cpp $(SIMPLUS)/SampST.h
	${CXX} $(FLAGS) -c -o sampst.o $(SIMPLUS)/SampST.cpp -I$(SIMPLUS)

image.o: $(SIMPLUS)/Image.cpp $(SIMPLUS)/Image.h
	${CXX} $(FLAGS) -c -o image.o $(SIMPLUS)/Image.cpp -I$(SIMPLUS)

weibull.o: $(SIMPLUS)/WeibullDist.cpp $(SIMPLUS)/WeibullDist.h
	${CXX} $(FLAGS) -c -o weibull.o $(SIMPLUS)/WeibullDist.cpp -I$(SIMPLUS)

uniform.o: $(SIMPLUS)/UniformDist.cpp $(SIMPLUS)/UniformDist.h
	${CXX} $(FLAGS) -c -o uniform.o $(SIMPLUS)/UniformDist.cpp -I$(SIMPLUS)

exponential.o: $(SIMPLUS)/ExponentialDist.cpp $(SIMPLUS)/ExponentialDist.h
	${CXX} $(FLAGS) -c -o exponential.o $(SIMPLUS)/ExponentialDist.cpp -I$(SIMPLUS)

rngfactory.o: $(SIMPLUS)/RNGFactory.cpp $(SIMPLUS)/RNGFactory.h
	${CXX} $(FLAGS) -c -o rngfactory.o $(SIMPLUS)/RNGFactory.cpp -I$(SIMPLUS)

rawrng.o: $(SIMPLUS)/RawRNG.cpp $(SIMPLUS)/RawRNG.h
	${CXX} $(FLAGS) -c -o rawrng.o $(SIMPLUS)/RawRNG.cpp -I$(SIMPLUS)

localrng.o: $(SIMPLUS)/LocalRNG.cpp $(SIMPLUS)/LocalRNG.h
	${CXX} $(FLAGS) -c -o localrng.o $(SIMPLUS)/LocalRNG.cpp -I$(SIMPLUS)

netrng.o: $(SIMPLUS)/NetRNG.cpp $(SIMPLUS)/NetRNG.h
	${CXX} $(FLAGS) -c -o netrng.o $(SIMPLUS)/NetRNG.cpp -I$(SIMPLUS)

filerng.o: $(SIMPLUS)/FileRNG.cpp $(SIMPLUS)/FileRNG.h
	${CXX} $(FLAGS) -c -o filerng.o $(SIMPLUS)/FileRNG.cpp -I$(SIMPLUS)

httpsocket.o: $(SIMPLUS)/HTTPSocket.cpp $(SIMPLUS)/HTTPSocket.h
	${CXX} $(FLAGS) -c -o httpsocket.o $(SIMPLUS)/HTTPSocket.cpp -I$(SIMPLUS)

rawdist.o: $(SIMPLUS)/RawDist.cpp $(SIMPLUS)/RawDist.h
	${CXX} $(FLAGS) -c -o rawdist.o $(SIMPLUS)/RawDist.cpp -I$(SIMPLUS)

.PHONY: clean count

clean:
	@rm -rf $(OBJS) main.o lab2

count:
	@wc -l *.cpp *.h

