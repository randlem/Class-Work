#include <iostream>
using std::cout;
using std::endl;

#include <vector>
using std::vector;

#include "../tableau.h"
#include "../fractional.h"

bool Fractional::output_float = true;

int main(int argc, char* argv[]) {
	Tableau t;
	int row,col;
	vector<Fractional> v;

	v.push_back(0);
	v.push_back(0);
	v.push_back(0);
	v.push_back(0);

	t.insertRow(v);

	v.clear();
	v.push_back(1);
	v.push_back(3);
	v.push_back(-4);
	v.push_back(0);

	t.insertRow(v);

	v.clear();
	v.push_back(0);
	v.push_back(-1);
	v.push_back(1);
	v.push_back(1);

	t.insertRow(v);

	v.clear();
	v.push_back(0);
	v.push_back(-1);
	v.push_back(2);
	v.push_back(4);

	t.insertRow(v);

	cout << t << endl;

	col = t.nextBasic();
	row = t.pivotRow(col);

	t.pivot(col,row);

	cout << t << endl;

	return 0;
}
