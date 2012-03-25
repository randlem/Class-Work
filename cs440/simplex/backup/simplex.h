#ifndef __SIMPLEX_H__
#define __SIMPLEX_H__

#define LESSER_EQUAL -1
#define EQUAL		  0
#define GREATER_EQUAL 1

#include <vector>
using std::vector;

#include <map>
using std::map;

#include <string>
using std::string;

#include "fractional.h"

typedef vector< Fractional > FracVector;

class Simplex {
public:
	Simplex(string filename) : cols(0), rows(0) {
		ofile.open(filename.c_str());
	}

	~Simplex() {
		ofile.close();
	}

	string getEnteringBasic() {
		return "";
	}

	int getExitingBasic() {
		return -1;
	}

	string buildVariable(string prefix, int col) {
		char buffer[10];
		sprintf(buffer, "%i", col);

		return prefix + buffer;
	}

	RowType buildIdentityCol(int row, Fractional coefficient) {
		int i;
		FracVector v;

		for(i=0; i < rows; ++i) {
			if(i == row)
				v.push_back(coefficient);
			else
				v.push_back(Fractional(0));
		}
	}

	int addArtifical(int row) {
		FracVector v = buildIdentityCol(row,Fractional(1));

		v[0] = 1; // set Big-M coefficient

		//

	}

	bool loadTableau(FracVector &objective_function, vector< FracVector > &constraints) {
		map<string,FracVector> t; // this will help to build the damn tableau
		map<string,FracVector>::iterator j;
		RowType rhs;
		vector<int> artifical;
		int col, k, row;

		cols = 0;
		rows = 2 + constraints.size();

		// build the objective function columns (including Big-M row)
		variables[cols] = "z"; // z column always first

		t[variables[cols]].push_back(Fractional(0)); // Big-M row z column
		t[variables[cols]].push_back(Fractional(1)); // objective row z column

		cols++;

		for (RowType::iterator i=objective_function.begin();
				i != objective_function.end(); ++i) {
			variables[cols] = buildVariable("x", cols);

			t[variables[cols]].push_back(Fractional(0));
			t[variables[cols]].push_back(Fractional((*i) * -1));

			cols++;
		}

		// add the Big-M and z row rhs value
		t["rhs"].push_back(Fractional(0));
		t["rhs"].push_back(Fractional(0));

		// make M the basic variable in row 0 and z basic in row 1
		basic[0] = "M";
		basic[1] = "z";

		// build the constraint rows (including extra variables)
		row = 2;
		for (vector< RowType >::iterator i=constraints.begin();
				i != constraints.end(); ++i) {
			t[variables[0]].push_back(Fractional(0)); // push the z row

			for(col=0; col < (*i).size() - 2; ++col)
				t[variables[col+1]].push_back((*i)[col]);

			switch ((*i)[(*i).size() - 2].getInt()) {
				case LESSER_EQUAL: { // add a slack variable
					variables[cols] = buildVariable("s", cols);
					t[variables[cols]] = buildIdentityCol(row, Fractional(1));
					basic[row] = variables[cols];
					cols++;
				} break;
				case GREATER_EQUAL: { // add an excess and artifical variable
					variables[cols] = buildVariable("e", cols);
					t[variables[cols]] = buildIdentityCol(row, Fractional(-1));
					cols++;
				} // fall through
				case EQUAL: { // add an artifical variable
					artifical.push_back(row);
				} break;
			}

			t["rhs"].push_back((*i)[(*i).size() - 1]);

			row++;
		}

		// do the artifical variables
		for (vector<int>::iterator i=artifical.begin(); i != artifical.end(); ++i) {
			variables[cols] = buildVariable("a", cols);
			addArtifical((*i));
			cols++;
		}

		// push the rhs symbol onto the variables stack
		variables[cols++] = "rhs";

		// push all the columns onto the tableau
		for (col = 0; col < cols; ++col) {
			tableau.insertColumn(t[variables[col]]);
		}

		return true;
	}

	bool solve() {
		// loop till we can't find another basic variable (i.e. no more negative
		// values in the Z row)
	}

private:
	map<string, FracVector> tableau;
	map<int,string> variables;
	map<int,string> basic;
	int cols, rows;

	ofstream ofile;
};

#endif
