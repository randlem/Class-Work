#include <iostream>
using std::cout;
using std::endl;

#include "simulation.h"

int main(int argc, char* argv[]) {
	Simulation sim(100000000, 1.0, 0.5);

	sim.run();
	sim.reportStats();

	return(0);
}
