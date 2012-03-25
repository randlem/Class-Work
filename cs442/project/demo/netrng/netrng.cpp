#include <iostream>
using std::cout;
using std::endl;

#include "SimPlus.h"

int main(int argc, char* argv[]) {
	NetRNG rng;
	int i = 0;

	while(1)
		cout << rng.genRandReal1() << endl;

	return(0);
}

