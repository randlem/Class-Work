#include "RawRNG.h"

const unsigned short RawRNG::Local=1;
const unsigned short RawRNG::Net=2;

RawRNG::RawRNG(unsigned short mode)
{
	RNGType=mode;

	lRNG=NULL;
	nRNG=NULL;

	if(Local==RNGType)
		lRNG = new LocalRNG;

	if(Net==RNGType)
		nRNG = new NetRNG;
}

RawRNG::~RawRNG()
{
	if(lRNG)
		delete lRNG;
	if(nRNG)
		delete nRNG;
}

double RawRNG::genRandReal1()
{
	if(lRNG)
		return lRNG->genRandReal1();

	if(nRNG)
		return nRNG->genRandReal1();
	return 0;
}

void RawRNG::seedRand(unsigned long seed)
{
	if(lRNG)
		lRNG->seedRand(seed);
}
