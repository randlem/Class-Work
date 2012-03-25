/**************************************************************

Program: SimPlus
Class:	 UniformDist
Group:	 Scott Harper, Tom Mancine, Ryan Scott

Description:
   The UniformDist class inherits from RawRNG which aggregates
   both the LocalRNG and NetRNG random number generators
   ( one pseudo and one true random number generator,
   respectively) for the interval [0,1].  Two parameters are
   received at construction, the upper and lower bound of the
   new uniformly distributed random number.  The getRandom()
   function will return one random double.

**************************************************************/

#include "RNGFactory.h"
#include "RawDist.h"

#ifndef UNIFORM_RNG_H
#define UNIFORM_RNG_H

class UniformDist : public RawDist {
	public:
		UniformDist(double a, double b, RNGFactory::RNGType = (RNGFactory::RNGType) 0);
		~UniformDist();
		double getRandom();
	protected:
		double lowerBound;
		double upperBound;
};

#endif
