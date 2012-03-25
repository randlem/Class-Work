/*****************************************************************

Program: SimPlus
Class:   NormalDist
Group:   Scott Harper, Tom Mancine, Ryan Scott

*****************************************************************/

#include <math.h>
#include "NormalDist.h"

/*
   The NormalDist constructor will receive 2 parameters,
   mu and sigma and set them as the mean and standard
   deviation for the normal random generator.  Other
   values are initialized.
*/
NormalDist::NormalDist( double mu, double sigma, RNGFactory::RNGType type ) : RawDist(type) {
	// set the mean and standard deviation at
	// construction time
	mean = mu;
	stdev = sigma;
	gotOne = false;
	hold = 0;
}

// Destructor: empty right now
NormalDist::~NormalDist() {

}

/*
   The getRandom() function will generate two normal random
   number according to the polar method.  The first number
   is returned and the second is held for the next getRandom()
   function call.
*/
double NormalDist::getRandom() {
	if(gotOne) {
		gotOne=false;
		return (hold * stdev) + mean;
	} else {
		double u1, u2, v1, v2, s, z1;
		do {
			// generate uniform numbers from LocalRNG
			u1 = rng->genRandReal1();
			u2 = rng->genRandReal1();

			v1 = (u1 * 2) - 1;
			v2 = (u2 * 2) - 1;

			s = (v1 * v1) + (v2 * v2);
		} while( s > 1 ); // value cannot be over 1

		z1 = v1 * pow( ((-2 * log(s))/s), 0.5 );
		hold = v2 * pow( ((-2 * log(s))/s), 0.5 );
		gotOne=true;

		return (z1 * stdev) + mean;
	}
}
