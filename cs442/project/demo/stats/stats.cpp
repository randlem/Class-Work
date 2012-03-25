#include <iostream>
using std::cout;
using std::endl;

#include "SampST.h"
#include "BinomialDist.h"
#include "PoissonDist.h"
#include "UniformDist.h"

int main() {
	SampST stat;
	BinomialDist bin(10,.5);
	PoissonDist pos(6.0);
	UniformDist unf(0.0,1.0);
	int counts[100];

	memset(counts,0,sizeof(int) * 100);

	for(int i=0; i < 100; ++i) {
		double b = unf.getRandom();
		//counts[(int)b]++;
		//cout << b << endl;
		stat.observe(i, b);
	}

//	for(int i=0; i < 100; ++i) {
//		cout << i << " " << counts[i] << endl;
//	}

//	cout << stat.getMean() << " " << stat.getStdDev() << endl;

	stat.writeCSV("test");
//	stat.writeHistogram("hist",1024,512,1);
	stat.writeVsTime("vstime",1024,512);

	return(0);
}
