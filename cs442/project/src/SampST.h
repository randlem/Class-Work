/****************************************************************************

SampST.h

Originally Written by: Scott Harper, Tom Mancine, Ryan Scott

SampST allows the user to specify handles to simple statistics-gathering
mechanisms.

METHODS:
--------

SampST()
Sole constructor.  Initializes data members.

~SampST()
Destructor.

observe(double, double)
Records the time (parm 1) and the data (parm 2) to a map, see if the data is either
a max or a minimum, and adds the data to the running sum.

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

getVariance() : double
Calculates and returns the current variance of the observations.

getStdDev() : double
Calculates and returns the current standard deviation of the observations.

writeCSV(string) : bool
Writes out a CSV (comma spaced) file for data import into a variety of
statistical and spreadsheet packages, such as MS Excel or OO.org Calc.
Parameters taken is the filename, sans extension, of the file to be written.
Outputted are the time (key) of the recorded sample, and the data associated
with that key.  Both are enclosed in double quotes, and lines are terminated
with a single carrage return (Unix style).  Returns true on successful file
creation.

writeHistogram(string, int, int, double) : bool
Create and write out a histogram of the data as thus observed.  Parameters taken
are a filename (sans extension), the width and height of the image in pixels,
and a double which sets the grain of the histogram ranges.  The range should be
a real number that makes sense for the data to be outputed.  This function may
behave abnormally shoud a bad range be inputted.  The histogram will currently
be saved as PNG file, using a default color scheme that's "easy" to see.  No
markings are made other then the drawing of the graph as the Image class does
not yet support the outputting of text.  X-Axis scale is determined by the grain
and the number of data partitions, with one tick per partition.  The Y-Axis
scale is dependent on the maximum count of elements per partition.  There will
always be a multiple of 10 tick marks on the axis.  The return value of the
function is determined on the success or failure of the Image classes file
writing method.

SampST::writeVsTime(string, int, int) : bool
Line plots the data as it was entered into the array.

****************************************************************************/

#include <vector>
using std::vector;

#include <map>
using std::map;

#include <string>
using std::string;

#include <fstream>
using std::ofstream;

#include <algorithm>
using std::max_element;

#include <iostream>
using std::cout;
using std::endl;

#include <math.h>

#include "Image.h"

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
		void observe(double,double);

		// basic getters
		double getSum()              { return sum; };
		double getMinimum()          { return minimum; };
		double getMaximum()          { return maximum; };
		unsigned int getSampleSize() { return samples.size(); };

		// calculated getters
		double getMean();
		double getStdDev();
		double getVariance();
		double getMedian();
		double getQ1();
		double getQ3();

		// file getters
		bool writeCSV(string);
		bool writeHistogram(string, int, int, double);
		bool writeBoxPlot(string, int, int);
		bool writeVsTime(string, int, int);

	protected:

	private:
		double sum;
		double minimum;
		double minimumT;
		double maximum;
		double maximumT;
		unsigned int sampleSize;

		map<double,double> samples;
};

#endif
