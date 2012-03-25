/****************************************************************************

SampST.cpp

Originally Written by: Scott Harper, Tom Mancine, Ryan Scott
Modified by: Mark Randles and Dan Sinclair

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
	minimumT = 0;
	maximum = 0;
	maximumT = 0;
}

SampST::~SampST()
{
}

void SampST::observe(double t, double observation)
{
	if( samples.size() == 0 )
	{
		minimum = observation;
		minimumT = t;
		maximum = observation;
		maximumT = t;
	}

	sum += observation;

	if(observation < minimum)
		minimum = observation;

	if(t < minimumT)
		minimumT = t;

	if(observation > maximum)
		maximum = observation;

	if(t > maximumT)
		maximumT = t;


	samples[t] = observation;
}

double SampST::getMean()
{
	if( samples.size() == 0 )
		return 0.0;
	return (sum / samples.size() );
}

double SampST::getVariance() {
	double sumSquared = 0.0;

	for(map<double,double>::iterator i=samples.begin(); i != samples.end(); ++i)
		sumSquared += (*i).second * (*i).second;

	return((sumSquared - ((sum * sum) / (double)samples.size())) / (double)(samples.size() - 1));
}

double SampST::getStdDev() {
	return(sqrt(getVariance()));
}

bool SampST::writeCSV(string fileName) {
	ofstream out;

	fileName += ".csv";
	out.open(fileName.c_str());
	if(!out)
		return(false);

	for(map<double,double>::iterator i=samples.begin(); i != samples.end(); ++i)
		out << "\"" << (*i).first << "\",\"" << (*i).second << "\"\n";

	out.close();

	return(true);
}

