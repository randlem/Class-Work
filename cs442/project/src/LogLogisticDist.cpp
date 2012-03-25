/*****************************************************************

Program: SimPlus
Class:	 LogLogisticDist
Group:	 Mark Randles, Dan Sinclair

*****************************************************************/
#include <math.h>

#include "LogLogisticDist.h"

/*
  The contructor receives two parameters N and P
*/
LogLogisticDist::LogLogisticDist( double a, double b, RNGFactory::RNGType type ) : RawDist(type) {
	A=a;
	B=b;
}

// The destructor: empty right now
LogLogisticDist::~LogLogisticDist() {

}


double LogLogisticDist::getRandom() {

	return( 1.0  / ( 1.0 + pow( rng->genRandReal1() / B , A) ) );
}
