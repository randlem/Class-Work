#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <iomanip>
using std::setprecision;

#include "FileRNG.h"
#include "LocalRNG.h"

const int CYCLES = 1000000000;
const int ENTRIES = 10;

int main() {
	FileRNG fRNG;
	LocalRNG lRNG;
	unsigned int i,j;
	int count,max,min;
	int* freq;
	int fileSize;
	double mean,mode;
	double sum;
	double variance;
	double n;
	unsigned int statCount[ENTRIES];

	cout << setprecision(20) << endl;

	fileSize = fRNG.getFileSize();
	cout << endl << fileSize << endl << endl;

	freq = new int[fileSize];
	memset(freq,0,sizeof(int) * fileSize);
	memset(statCount,0,sizeof(unsigned int) * ENTRIES);

	sum = 0.0;
	//for(i=0; i < CYCLES; i++) {
	while(1) {
		n = fRNG.genRandReal1();
		sum += n;
	//	freq[(int)n]++;
		if(n > 0 && n <= 0.1)
			statCount[0]++;
		else if(n > 0.1 && n <= 0.2)
			statCount[1]++;
		else if(n > 0.2 && n <= 0.3)
			statCount[2]++;
		else if(n > 0.3 && n <= 0.4)
			statCount[3]++;
		else if(n > 0.4 && n <= 0.5)
			statCount[4]++;
		else if(n > 0.5 && n <= 0.6)
			statCount[5]++;
		else if(n > 0.6 && n <= 0.7)
			statCount[6]++;
		else if(n > 0.7 && n <= 0.8)
			statCount[7]++;
		else if(n > 0.8 && n <= 0.9)
			statCount[8]++;
		else if(n > 0.9 && n <= 1.0)
			statCount[9]++;

		if(n < 0 || n > 1.0) {
			cout << i << " " << n << endl;
		}
		if(i > CYCLES) { cout << ++j << " " << (sum / (double)(j * CYCLES)) << endl; i = 0; }
		else ++i;
	}

	for(i=0; i < ENTRIES; ++i)
		cout << i << " " << statCount[i] << endl;

	cout << "Mean = " << (sum / (double)CYCLES) << endl;

	/*count = 0; max = freq[0]; min = 10000;
	for(i-0; i < fileSize; i++) {
		if(freq[i] > 0) {
			sum += freq[i];
			++count;
		}
		if(freq[i] > max)
			max = freq[i];
		if(freq[i] < min)
			min = freq[i];
	}

	mean = sum / (double)count;
	sum = 0;
	for(i=0; i < fileSize; i++) {
		sum += (double)((freq[i] - mean) * (freq[i] - mean));
		//cerr << "\"" << i << "\",\"" << freq[i] << "\"" << endl;
	}
	variance = sum / fileSize;

	cout << "Mean = " << mean << endl << "Variance = " << variance << endl << "Std. Dev. = " << sqrt(variance) << endl;
	cout << "Max = " << max << endl << "Min = " << min << endl;
	*/
	delete [] freq;

	return(0);
}

