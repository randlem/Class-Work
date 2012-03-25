/*****************************************************************

Program: SimPlus
Class:	 BinomialDist
Group:	 Mark Randles, Dan Sinclair

*****************************************************************/
#include <math.h>
#include "BinomialDist.h"

/*
  The contructor receives two parameters Num and P
*/
BinomialDist::BinomialDist( double a, double b, RNGFactory::RNGType type ) : RawDist(type) {
	Num=a;
	P=b;
}

// The destructor: empty right now
BinomialDist::~BinomialDist() {

}

double BinomialDist::getRandom() {

	return pow( ( 1 - P) + ( P * rng->genRandReal1() ), Num );
}
