FLAGS = -Wno-deprecated -g

LIBS = main.o

simplex: $(LIBS)
	$(CXX) $(CXXFLAGS) $(FLAGS) $(LIBS) -o simplex

fractional: fractional.h test/fractional.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) test/fractional.cpp -o fractional

tableau: tableau.h fractional.h test/tableau.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) test/tableau.cpp -o tableau

%.o: %.cpp fractional.h tableau.h simplex.h
	$(CXX) $(CXXFLAGS) $(FLAGS) -c $< -o $@



.PHONY: clean
# clean the object files, and any executeables present
clean:
	@rm -f *.o simplex fractional tableau
