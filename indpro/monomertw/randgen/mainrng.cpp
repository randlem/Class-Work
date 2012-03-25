#include <iostream>
using std::cout;
using std::endl;

#include "randgen.h"

int main() {
	RandGen rng(10);

	for(int i=0; i < 50; ++i) {
		cout << i << " " <<  rng.getRandom(i) << endl;
	}
	cout << " ------------------------------ " << endl;
	rng.rewind((double)25);
	for(int i=0; i < 25; ++i) {
		cout << i+25 << " " << rng.getRandom(i) << endl;
	}

	return(0);
}
