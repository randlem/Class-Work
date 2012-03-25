/*****************************************************************

Program: SimPlus
Class:	 BernoulliDist
Group:	 Mark Randles, Dan Sinclair

*****************************************************************/
#include <math.h>
#include "BernoulliDist.h"

/*
  The contructor receives two parameters Num and P
*/
BernoulliDist::BernoulliDist( double p, RNGFactory::RNGType type ) : RawDist(type) {
	this->p=p;
}

// The destructor: empty right now
BernoulliDist::~BernoulliDist() {

}

double BernoulliDist::getRandom() {
	return((rng->genRandReal1() <= p) ? 1.0 : 0.0);
}
