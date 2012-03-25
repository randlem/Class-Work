/****************************************************************************

RawDist.h

Originally Written by: Mark Randles and Dan Sinclair

Provides an abstract base class for all RNG distributions that will be written.

METHODS:
--------
RawDist()
Default constructor, sets the default RNG type.

~RawDist()
Deletes the RNG associated with the dist.

getRandom() : double
Pure virtual functions.  Will return the double random variate in derived classes.

****************************************************************************/

#include "RNGFactory.h"

#ifndef RAWDIST
#define RAWDIST

class RawDist {
	public:
		RawDist(RNGFactory::RNGType type) : rng(NULL) {
			rng = factory.getRNG(type);
		}
		~RawDist() {
			if(rng != NULL)
				delete rng;
		}
		virtual double getRandom() = 0;
	protected:
		RawRNG* rng;
		static RNGFactory factory;
};

#endif
