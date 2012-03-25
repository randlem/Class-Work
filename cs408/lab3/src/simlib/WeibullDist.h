/***********************************************************

Program: SimPlus
Class:	 WeibullDist
Group:   Mark Randles, Dan Sinclair

Description:
   BLAH BLAH

***********************************************************/

#include "RNGFactory.h"
#include "RawDist.h"

#ifndef WEI_Dist_H
#define WEI_Dist_H

class WeibullDist : public RawDist {
   public:
		WeibullDist(double, double, RNGFactory::RNGType);
		~WeibullDist();
		double getRandom( );
   protected:
		double A;
		double B;
};

#endif
