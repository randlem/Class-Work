/***********************************************************

Program: SimPlus
Class:	 LoglogisticDist
Group:   Mark Randles, Dan Sinclair

Description:
   BLAH BLAH

***********************************************************/

#include "RNGFactory.h"
#include "RawDist.h"

#ifndef LLO_Dist_H
#define LLO_Dist_H

class LogLogisticDist : public RawDist {
   public:
		LogLogisticDist(double, double, RNGFactory::RNGType);
		~LogLogisticDist();
		double getRandom( );
   protected:
		double A;
		double B;
};

#endif
