/***********************************************************

Program: SimPlus
Class:	 ExponentialDist
Group:   Scott Harper, Tom Manice, Ryan Scott

Description:
   The ExponentialDist class inherits from RawDist which
   aggregrates LocalRNG and NetRNG that are both uniform
   random number generators( 1 pseudo and 1 true).  It
   receives 1argument, theta, which is essentially the mean
   of the exponential distribution we would like to use for our
   random number generation.  The getRandom() function will
   return a random double that is from an exponential
   distribution with the specified mean.

***********************************************************/

#include "RNGFactory.h"
#include "RawDist.h"

#ifndef EXP_Dist_H
#define EXP_Dist_H

class ExponentialDist : public RawDist {
   public:
		ExponentialDist(double, RNGFactory::RNGType = (RNGFactory::RNGType) 0);
		~ExponentialDist();
		double getRandom( );
   protected:
		double theta;
};

#endif
