/***********************************************************

Program: SimPlus
Class:	 PoissonDist
Group:   Mark Randles, Dan Sinclair

Description:
   BLAH BLAH

***********************************************************/

#include "RNGFactory.h"
#include "RawDist.h"

#ifndef POI_Dist_H
#define POI_Dist_H

class PoissonDist : public RawDist {
   public:
		PoissonDist(double, RNGFactory::RNGType = (RNGFactory::RNGType) 0);
		~PoissonDist();
		double getRandom( );
   protected:
		double lambda;
		double a;
		double b;
		double i;
};

#endif
