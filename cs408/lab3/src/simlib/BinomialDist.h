/***********************************************************

Program: SimPlus
Class:	 BinomialDist
Group:   Mark Randles, Dan Sinclair

Description:
   BLAH BLAH

***********************************************************/

#include "RNGFactory.h"
#include "RawDist.h"

#ifndef BIN_DIST_H
#define BIN_DIST_H

class BinomialDist : public RawDist {
   public:
		BinomialDist(double, double, RNGFactory::RNGType = (RNGFactory::RNGType) 0);
		~BinomialDist();
		double getRandom( );
   protected:
		double t;
		double p;
};

#endif
