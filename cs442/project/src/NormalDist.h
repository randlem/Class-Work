/*********************************************************************
Program: SimPlus
Class:	 NormalDist
Group:	 Tom Mancine, Ryan Scott, Scott Harper

Description:
   This class inherits from RawRNG, which aggregates both LocalRNG
   and NetRNG uniform random number generators for the interval [0,1].
   It receives two arguments at construction, mu and sigma (mean and
   standard deviation respectively). A polar method is used to
   generate two new random numbers (one is saved) that conformto a
   normal distribution. The getRandom() function will return one
   normal random number.


********************************************************************/

#include "RNGFactory.h"
#include "RawDist.h"

#ifndef NORMAL_RNG_H
#define NORMAL_RNG_H

class NormalDist : public RawDist {
	public:
		NormalDist( double, double, RNGFactory::RNGType);
		~NormalDist();
		double getRandom();
	protected:
		double mean;
		double stdev;
		double hold;
		bool gotOne;
};

#endif
