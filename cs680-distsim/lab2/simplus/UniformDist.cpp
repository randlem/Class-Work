/*****************************************************************

Program: SimPlus
Class:   UniformDist
Group:   Scott Harper, Tom Mancine, Ryan Scott

*****************************************************************/

#include "UniformDist.h"

/*
   The contructor receives 2 parameters which are set as the
   lower and upper bounds for the uniform random number
   generator.
*/
UniformDist::UniformDist( double a, double b, RNGFactory::RNGType type) : RawDist(type) {
	lowerBound = a;
	upperBound = b;
}

/*
   The destructor: empty now
*/
UniformDist::~UniformDist() {

}

/*
   The getRandom() function uses LocalRNG's uniform random
   number generator, and scales/tranlates it up to the specified
   boundaries of the new uniform random number generator.  The
   new number is returned.
*/
double UniformDist::getRandom( ) {
   return(lowerBound + (rng->genRandReal1() * (upperBound - lowerBound)));
}
