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

bool SampST::writeHistogram(string fileName, int width, int height, double rangeGrain) {
	int rangeCount = ((maximum - minimum) / rangeGrain);
	vector<int> counts(rangeCount);
	Image img(width,height,fileName);
	double aspect = width / height;
	int wMargin = width / (10 * ((aspect < 1.0) ? (1.0 / aspect) : 1.0));
	int hMargin = height / (10 * ((aspect > 1.0) ? (1.0 / aspect) : 1.0));
	int wAxisLen = width - (2 * wMargin);
	int hAxisLen = height - (2 * hMargin);
	int tickOffset = 2;
	int offset = 0;
	int hAxisTick = 1;
	int countMax = 0;
	point p1;
	point p2;
	rgb black = {  0,   0,   0};
	rgb white = {255, 255, 255};
	rgb blue  = {  0,   0, 255};

	// gather the histogram counts
	for(map<double,double>::iterator i=samples.begin(); i != samples.end(); ++i) {
		int j = (*i).second / rangeGrain;
		counts[j]++;
	}

	// DEBUGGING
	//cout << rangeCount << " " << maximum << " " << minimum << endl;
	//for(int i=0; i < rangeCount; ++i)
	//	cout << i << " " << counts[i] << endl;
	//cout << endl;

	// MORE DEBUGGING
	//cout << endl << "Image Stats: " << endl;
	//cout << "w=" << width << " h=" << height << " aspect=" << aspect << " 1/aspect=" << (1.0 / aspect) << endl;
	//cout << "wMargin=" << wMargin << "(" << ((aspect < 1.0) ? (1.0 / aspect) : 1.0) << ") hMargin=" << hMargin << "(" << ((aspect > 1.0) ? (1.0 / aspect) : 1.0) << ")" << endl;
	//cout << "max=" << *max_element(counts.begin(),counts.end()) << endl;

	// blank out the image to a good color
	img.clearImage(white);

	// draw the horozontal axis line
	p1.x = wMargin;
	p1.y = height - hMargin;
	p2.x = width - wMargin;
	p2.y = height - hMargin;
	img.drawLine(p1,p2,black);

	// draw the verticle axis line
	p1.x = wMargin;
	p1.y = hMargin;
	p2.x = wMargin;
	p2.y = height - hMargin;
	img.drawLine(p1,p2,black);

	// draw the verticle tickmarks, should be ten every time
	countMax = *max_element(counts.begin(),counts.end());
	hAxisTick = 1;
	while((countMax/=10) >= 10.0)
		hAxisTick *= 10;
	countMax = *max_element(counts.begin(),counts.end());
	offset = (hAxisLen * hAxisTick) / countMax;
	for(int i=offset; i <= hAxisLen; i += offset) {
		p1.x = wMargin + tickOffset;
		p1.y = i + hMargin;
		p2.x = wMargin - tickOffset;
		p2.y = i + hMargin;
		img.drawLine(p1,p2,black);
	}

	// YET MORE DEBUGGING
	//cout << "hAxisTick=" << hAxisTick << " countMax=" << countMax << endl;

	// draw the histogram boxes
	offset = wAxisLen / rangeCount;
	countMax = *max_element(counts.begin(),counts.end());
	for(int i=0; i < rangeCount; ++i) {
		p1.x = (i * offset) + wMargin;
		p1.y = height - hMargin;
		p2.x = ((i + 1) * offset) + wMargin;
		p2.y = (height - hMargin) - (int)((double)hAxisLen * (double)((double)counts[i] / (double)countMax));
		img.drawRect(p1,p2,blue);
		p1.x = (i * offset) + wMargin;
		p1.y = (height - hMargin) - (int)((double)hAxisLen * (double)((double)counts[i] / (double)countMax));
		p2.x = ((i + 1) * offset) + wMargin;
		p2.y = (height - hMargin) - (int)((double)hAxisLen * (double)((double)counts[i] / (double)countMax));
		img.drawLine(p1,p2,black);
	}

	// draw the tickmarks on the horozontal axis, one for each rangeCount
	offset = wAxisLen / rangeCount;
	for(int i=0; i < rangeCount; ++i) {
		p1.x = (i * offset) + wMargin;
		p1.y = height - hMargin + tickOffset;
		p2.x = (i * offset) + wMargin;
		p2.y = (height - hMargin) - (int)((double)hAxisLen * (double)((double)counts[i] / (double)countMax));
		img.drawLine(p1,p2,black);
		p1.x = ((i+1) * offset) + wMargin;
		p1.y = height - hMargin + tickOffset;
		p2.x = ((i+1) * offset) + wMargin;
		p2.y = (height - hMargin) - (int)((double)hAxisLen * (double)((double)counts[i] / (double)countMax));
		img.drawLine(p1,p2,black);
	}

	// output the image
	return(img.outputImage());
}

bool SampST::writeVsTime(string fileName, int width, int height) {
	Image img(width,height,fileName);
	double aspect = width / height;
	int wMargin = width / (10 * ((aspect < 1.0) ? (1.0 / aspect) : 1.0));
	int hMargin = height / (10 * ((aspect > 1.0) ? (1.0 / aspect) : 1.0));
	int wAxisLen = width - (2 * wMargin);
	int hAxisLen = height - (2 * hMargin);
	int tickOffset = 2;
	int offset = 0;
	int axisTick = 1;
	double observationRange = maximum - minimum;
	double timeRange = maximumT - minimumT;
	double temp = 0.0;
	point p1;
	point p2;
	point p3;
	rgb black = {  0,   0,   0};
	rgb white = {255, 255, 255};
	rgb blue  = {  0,   0, 255};

	// blank out the image to a good color
	img.clearImage(white);

	// draw the horozontal axis line
	p1.x = wMargin;
	p1.y = height - hMargin;
	p2.x = width - wMargin;
	p2.y = height - hMargin;
	img.drawLine(p1,p2,black);

	// draw the verticle axis line
	p1.x = wMargin;
	p1.y = hMargin;
	p2.x = wMargin;
	p2.y = height - hMargin;
	img.drawLine(p1,p2,black);

	// draw the verticle tickmarks, should be ten every time
	axisTick = 1;
	temp = observationRange;
	while((temp/=10) >= 1.0)
		axisTick *= 10;
	offset = (hAxisLen * axisTick) / observationRange;
	for(int i=offset; i <= hAxisLen; i += offset) {
		p1.x = wMargin + tickOffset;
		p1.y = i + hMargin;
		p2.x = wMargin - tickOffset;
		p2.y = i + hMargin;
		img.drawLine(p1,p2,black);
	}

	// draw the tickmarks on the horozontal axis, one for each rangeCount
	axisTick = 1;
	temp = timeRange;
	while((temp/=10) >= 10.0)
		axisTick *= 10;
	offset = (wAxisLen * axisTick) / timeRange;
	for(int i=offset; i <= wAxisLen; i += offset) {
		p1.x = i + wMargin;
		p1.y = height - hMargin + tickOffset;
		p2.x = i + wMargin;
		p2.y = height - hMargin - tickOffset;
		img.drawLine(p1,p2,black);
	}

	p2.x = wMargin;
	p2.y = height - hMargin;
	for(map<double,double>::iterator i=samples.begin(); i != samples.end(); ++i) {
		p1.x = p2.x;
		p1.y = p2.y;
		p2.x = wMargin + (int)((double)wAxisLen * (double)((double)(*i).first / (double)timeRange));
		p2.y = (height - hMargin) - (int)((double)hAxisLen * (double)((double)(*i).second / (double)observationRange));
		img.drawLine(p1,p2,black);
	}

	// output the image
	return(img.outputImage());
}
