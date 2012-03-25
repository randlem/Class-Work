#include <iostream>
using std::cout;
using std::endl;

#include "genrand_cpp.h"

int main() {
	int i;
	RandGenEngine rge;
		
	for(i=0; i < 20; i++)
		cout << i << ": " << rge.randNum() << endl;

	return(0);
}
