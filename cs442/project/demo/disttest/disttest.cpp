#include <iostream>
using std::cout;
using std::endl;

#include "PoissonDist.h"
#include "BinomialDist.h"

int main() {
	PoissonDist pd(1.0);
	BinomialDist bd(1.0,1.0);
	int i=0;

	for(i=0; i < 10; i++)
		cout << pd.getRandom() << endl;

	cout << endl;

	for(i=0; i < 10; i++)
		cout << bd.getRandom() << endl;


	return(0);
}