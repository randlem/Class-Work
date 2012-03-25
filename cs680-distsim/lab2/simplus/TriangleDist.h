/***********************************************************

Program: SimPlus
Class:	 TriangleDist
Group:   Mark Randles, Dan Sinclair

Description:
   BLAH BLAH

***********************************************************/

#include "RNGFactory.h"
#include "RawDist.h"

#ifndef TRI_Dist_H
#define TRI_Dist_H

class TriangleDist : public RawDist {
   public:
		TriangleDist(double, double, double, RNGFactory::RNGType);
		~TriangleDist();
		double getRandom( );
   protected:
		double A;
		double B;
		double C;
};

#endif
