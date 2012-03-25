#include "LocalRNG.h"
#include "NetRNG.h"
#include "FileRNG.h"
#include "RawRNG.h"

#ifndef RNGFACTORY
#define RNGFACTORY

class RNGFactory {
	public:
		enum RNGType {Local, Net, File};

		RawRNG* getRNG(RNGType);
	private:
};

#endif
