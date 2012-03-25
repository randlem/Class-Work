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
