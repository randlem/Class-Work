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
BinomialDist::BinomialDist( double t, double p, RNGFactory::RNGType type ) : RawDist(type) {
	this->t=t;
	this->p=p;
}

// The destructor: empty right now
BinomialDist::~BinomialDist() {

}

double BinomialDist::getRandom() {
	double X = 0;

	for(int i=0; i < t; ++i)
		X += (rng->genRandReal1() <= p) ? 1 : 0;

	return(X);
}
