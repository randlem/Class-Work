/*****************************************************************

Program: SimPlus
Class:   UniformRNG
Group:   Scott Harper, Tom Mancine, Ryan Scott

*****************************************************************/

#include "UniformRNG.h"
#include "LocalRNG.h"

/*
   The contructor receives 2 parameters which are set as the
   lower and upper bounds for the uniform random number
   generator.
*/
UniformRNG::UniformRNG( double a, double b, unsigned short mode)
: RawRNG(mode)
{
   lowerBound = a;
   upperBound = b;
}

/*
   The destructor: empty now
*/
UniformRNG::~UniformRNG()
{

}

/*
   The getRandom() function uses LocalRNG's uniform random
   number generator, and scales/tranlates it up to the specified
   boundaries of the new uniform random number generator.  The
   new number is returned.
*/
double UniformRNG::getRandom( )
{
   return lowerBound + ( genRandReal1() * (upperBound - lowerBound) );
}
