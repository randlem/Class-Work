/**************************************************************

Program: SimPlus
Class:	 UniformRNG
Group:	 Scott Harper, Tom Mancine, Ryan Scott

Description:
   The UniformRNG class inherits from RawRNG which aggregates
   both the LocalRNG and NetRNG random number generators
   ( one pseudo and one true random number generator,
   respectively) for the interval [0,1].  Two parameters are
   received at construction, the upper and lower bound of the
   new uniformly distributed random number.  The getRandom() 
   function will return one random double.

**************************************************************/

#ifndef UNIFORM_RNG_H
#define UNIFORM_RNG_H

#include "RawRNG.h"

class UniformRNG : public RawRNG
   {
   public:
      UniformRNG( double a, double b, unsigned short mode = 
RawRNG::Local );
      ~UniformRNG();
      double getRandom();
   protected:
      double lowerBound;
      double upperBound;

   };

#endif
