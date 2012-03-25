#include <iostream>
using std::cout;
using std::endl;

#include "../fractional.h"

int main (int varc, char* argv[]) {
	Fractional a(3,4),b(1,4),c(1,2),d(3,2),e(1,5),f("3/2"),g("12");

	if (a > b)
		cout << "true" << endl;
	else
		cout << "false" << endl;

	if (b > c)
		cout << "true" << endl;
	else
		cout << "false" << endl;

	if (d > a)
		cout << "true" << endl;
	else
		cout << "false" << endl;

	if (a > c)
		cout << "true" << endl;
	else
		cout << "false" << endl;



	b = b + 1;
	d = d - 1;
	c = c * 2;
	a = a / 3;

	cout << b << endl;
	cout << d << endl;
	cout << c << endl;
	cout << a << endl;
	cout << f << endl;
	cout << g << endl;

	return 0;
}

