/***********************************************************

Program: SimPlus
Class:	 ExponentialRNG
Group:   Scott Harper, Tom Manice, Ryan Scott

Description:
   The ExponentialRNG class inherits from RawRNG which
   aggregrates LocalRNG and NetRNG that are both uniform 
   random number generators( 1 pseudo and 1 true).  It
   receives 1argument, theta, which is essentially the mean
   of the exponential distribution we would like to use for our
   random number generation.  The getRandom() function will
   return a random double that is from an exponential 
   distribution with the specified mean.

***********************************************************/


#ifndef EXP_RNG_H
#define EXP_RNG_H

#include "RawRNG.h"

class ExponentialRNG : public RawRNG
   {
   public:
      ExponentialRNG( double, unsigned short = RawRNG::Local );
      ~ExponentialRNG();
      double getRandom( );
   protected:
      double theta;
   };

#endif
