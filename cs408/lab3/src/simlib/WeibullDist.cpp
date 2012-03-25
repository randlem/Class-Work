/*****************************************************************

Program: SimPlus
Class:	 WeibullDist
Group:	 Mark Randles, Dan Sinclair

*****************************************************************/
#include <math.h>

#include "WeibullDist.h"

/*
  The contructor receives two parameters N and P
*/
WeibullDist::WeibullDist( double a, double b, RNGFactory::RNGType type ) : RawDist(type) {
	A=a;
	B=b;
}

// The destructor: empty right now
WeibullDist::~WeibullDist() {

}


double WeibullDist::getRandom() {
   
	return 1 - exp( -1.0 * pow( rng->genRandReal1() / B , A ) );
}
