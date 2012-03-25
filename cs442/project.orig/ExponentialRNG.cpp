/*****************************************************************

Program: SimPlus
Class:	 ExponentialRNG
Group:	 Scott Harper, Tom Mancine, Ryan Scott

*****************************************************************/
#include <math.h>

#include "ExponentialRNG.h"
#include "LocalRNG.h"

/*
  The contructor receives one parameter which is essentially
  the mean of the exponential distribution.
*/
ExponentialRNG::ExponentialRNG( double a, unsigned short mode )
: RawRNG(mode)
{
   theta=a;
}

// The destructor: empty right now
ExponentialRNG::~ExponentialRNG()
{

}

/*
   The getRandom() function uses LocalRNG's random number
   generator and uses that number to return a new exponentially
   distributed random number according to the mean of the
   distribition.
*/
double ExponentialRNG::getRandom( )
{
   return ( (-1*theta) * log(genRandReal1()) );
}
