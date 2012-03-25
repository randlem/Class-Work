LIBS = -lpthread

ddr: ddr.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -o ddr ddr.o $(LIBS)

ddr.o: ddr.cpp ddr.h
	$(CXX) $(CXXFLAGS) $(FLAGS) -c -o ddr.o ddr.cpp

.PHONY: clean
clean:
	@rm -f *.o ddr
