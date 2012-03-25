/***********************************************************

Program: SimPlus
Class:	 BernoulliDist
Group:   Mark Randles, Dan Sinclair

Description:
   BLAH BLAH

***********************************************************/

#include "RNGFactory.h"
#include "RawDist.h"

#ifndef BER_DIST_H
#define BER_DIST_H

class BernoulliDist : public RawDist {
   public:
		BernoulliDist(double, RNGFactory::RNGType = (RNGFactory::RNGType) 0);
		~BernoulliDist();
		double getRandom( );
   protected:
		double p;
};

#endif
