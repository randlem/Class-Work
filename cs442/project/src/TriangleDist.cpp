/*****************************************************************

Program: SimPlus
Class:	 TriangleDist
Group:	 Mark Randles, Dan Sinclair

*****************************************************************/
#include <math.h>

#include "TriangleDist.h"

/*
  The contructor receives two parameters N and P
*/
TriangleDist::TriangleDist( double a, double b, double c, RNGFactory::RNGType type ) : RawDist(type) {
	A=a;
	B=b;
	C=c;
}

// The destructor: empty right now
TriangleDist::~TriangleDist() {

}


double TriangleDist::getRandom() {

	double x = rng->genRandReal1();

	if( x < A )
		return 0;

	if( A <= x && x <= C )
		return ( ( x - A ) * ( x - A ) ) / ( ( B - A ) * ( C - A ) );

	if( C < x && x <= B )
		return 1.0 - ( ( B - x ) * ( B - x ) ) / ( ( B - A ) * ( B - C ) );

	return 1;
}
