#include <iostream>
using std::cout;
using std::endl;

#include "histogram.h"

int main(int argc, char* argv[]) {
	Histogram a(10),b(10),c(10);

	srand(1337);

	for(int i=0; i < 10; i++)
		a[i] = rand() % 10;

	cout << a.toString() << endl
	     << b.toString() << endl
		 << c.toString() << endl << endl;

	c = b = a;
	c[0]++;
	c[1] = a[1] + b[1];

	cout << a.toString() << endl
	     << b.toString() << endl
		 << c.toString() << endl << endl;

	b -= a;
	c += a;

	cout << a.toString() << endl
	     << b.toString() << endl
		 << c.toString() << endl << endl;

	b = a;
	c = b - a;

	cout << a.toString() << endl
	     << b.toString() << endl
		 << c.toString() << endl << endl;

	b = a;
	c = b + a;

	cout << a.toString() << endl
	     << b.toString() << endl
		 << c.toString() << endl << endl;

	cout << "c.min() = " << c[c.min()]
	     << " c.min() = " << c[c.max()] << endl;;

	return 0;
}
