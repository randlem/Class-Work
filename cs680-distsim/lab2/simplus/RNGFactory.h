/****************************************************************************

RNGFactory.cpp

Originally Written by: Mark Randles and Dan Sinclair

Provides a psudo-factory for RNG classes.  Uses an enum type to get determine what
RNG to return.

METHODS:
--------
getRNG(RNGType) : RawRNG*
Returns a pointer to a RawRNG derived RNG object depending on the enum type passed in.

****************************************************************************/
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
