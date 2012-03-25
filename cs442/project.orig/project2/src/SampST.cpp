/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

SampST.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of SampST,
see SampST.h.

****************************************************************************/

#include "SampST.h"

SampST::SampST()
{
	sum = 0.0;
	sampleSize = 0;
	minimum = 0;
	maximum = 0;
}

SampST::~SampST()
{
}

void SampST::observe(double observation)
{
	if( sampleSize == 0 )
	{
		minimum = observation;
		maximum = observation;
	}

	++sampleSize;

	sum += observation;

	if(observation < minimum)
		minimum = observation;

	if(observation > maximum)
		maximum = observation;
}

double SampST::getMean()
{
	if( sampleSize == 0 )
		return 0.0;
	return (sum / sampleSize);
}
