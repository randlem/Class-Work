/*****************************************************************

Program: SimPlus
Class:	 ExponentialDist
Group:	 Scott Harper, Tom Mancine, Ryan Scott

*****************************************************************/
#include <math.h>

#include "ExponentialDist.h"

/*
  The contructor receives one parameter which is essentially
  the mean of the exponential distribution.
*/
ExponentialDist::ExponentialDist( double a, RNGFactory::RNGType type ) : RawDist(type) {
	theta=a;
}

// The destructor: empty right now
ExponentialDist::~ExponentialDist() {

}

/*
   The getRandom() function uses LocalRNG's random number
   generator and uses that number to return a new exponentially
   distributed random number according to the mean of the
   distribition.
*/
double ExponentialDist::getRandom() {
   return((-1*theta) * log(rng->genRandReal1()));
}
