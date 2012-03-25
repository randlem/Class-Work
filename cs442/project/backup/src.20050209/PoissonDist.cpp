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
PoissonDist::PoissonDist( double a, RNGFactory::RNGType type ) : RawDist(type) {
	lambda=a;
}

// The destructor: empty right now
PoissonDist::~PoissonDist() {

}

double PoissonDist::getRandom() {
		return exp( lambda * ( rng->genRandReal1() - 1 ) );
}
