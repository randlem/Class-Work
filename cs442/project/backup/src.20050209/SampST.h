/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

SampST.h

SampST allows the user to specify handles to simple statistics-gathering
mechanisms.

METHODS:
--------

SampST()
Sole constructor.  Initializes data members.

~SampST()
Destructor.

observe(double)
Adds its sole argument to the observed sum and increments the number of
observations.  Tracks the smallest and largest observations.

getSum() : double
Returns the current sum of all observations.

getMinimum() : double
Returns the current minimum of all observations.

getMaximum() : double
Returns the current maximum of all observations.

getSampleSize() : unsigned int
Returns the current number of observations.

getMean() : double
Calculates and returns the current mean of the observations.

****************************************************************************/
#ifndef SAMPLE_H
#define SAMPLE_H

class SampST
{

	public:
		// Constructors
		SampST();

		// Destructor
		~SampST();

		// Take a sample
		void observe(double);

		// basic getters
		double getSum()              { return sum; };
		double getMinimum()          { return minimum; };
		double getMaximum()          { return maximum; };
		unsigned int getSampleSize() { return sampleSize; };

		// calculated getters
		double getMean();

	protected:

	private:
		double sum;
		double minimum;
		double maximum;
		unsigned int sampleSize;
};

#endif
