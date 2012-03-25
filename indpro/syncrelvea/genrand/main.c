#include <stdio.h>

#include "genrand.h"

int main() {
	int i;
		
	/* start the random generator engine */
	initRandGenEngine();

	for(i=0; i < 20; i++)
		printf("%i: %f\n",i,randNum());

	/* shutdown the random generator engine */
	shutdownRandGenEngine();

	return(0);
}
