/****************************************************************************

RNGFactory.cpp

Originally Written by: Mark Randles and Dan Sinclair

For more information see RNGFactory.h

****************************************************************************/
#include "RNGFactory.h"

RawRNG* RNGFactory::getRNG(RNGType type) {
	switch(type) {
		case Local: {
			return(new LocalRNG());
		} break;
		case Net: {
			return(new NetRNG());
		} break;
		case File: {
			return(new FileRNG());
		} break;
	}
	return(NULL);
}

