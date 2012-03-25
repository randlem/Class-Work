/*****************************************************************

Program: SimPlus
Class:	 PoissonDist
Group:	 Mark Randles, Dan Sinclair

*****************************************************************/
#include <math.h>
#include "PoissonDist.h"

/*
  The contructor receives one parameter which is lambda, the mean of the poisson distribution
*/
PoissonDist::PoissonDist( double l, RNGFactory::RNGType type ) : RawDist(type) {
	lambda=l;
	a = pow(M_E,-lambda);
	b = 1;
	i = 0;
}

// The destructor: empty right now
PoissonDist::~PoissonDist() {

}

double PoissonDist::getRandom() {
	i = -1;
	b = 1;

	do {
		b = b * rng->genRandReal1();
		++i;
	} while(b >= a);

	return(i);
	//	return exp( lambda * ( rng->genRandReal1() - 1 ) );
}
