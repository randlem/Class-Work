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
	Simplex(string filename) : cols(0), rows(0), bigM(false) {
		ofile.open(filename.c_str());
	}

	~Simplex() {
		ofile.close();
	}

	string getEnteringBasic() {
		string basic = "";

		if (bigM) {
			Fractional min_bigM(0), min(0);

			for (vector<string>::iterator i=variables.begin();
					i != variables.end(); ++i) {
				if(*i == "z" || *i == "rhs")
					continue;

				if (tableau[*i][0] == min_bigM) {
					if(tableau[*i][1] < min) {
						min = tableau[*i][1];
						basic = *i;
					}
				} else if (tableau[*i][0] < min) {
					min = tableau[*i][0];
					basic = *i;
				}
			}

		} else {
			Fractional min(0);

			for (vector<string>::iterator i=variables.begin();
					i != variables.end(); ++i) {
				if(*i == "z")
					continue;

				if (tableau[*i][1] < min) {
					min = tableau[*i][1];
					basic = *i;
				}
			}

		}

		return basic;
	}

	int getExitingBasic(string col) {
		int i, row;
		Fractional min_ratio(0);

		// set the initial entering basic row to the first positive non-zero
		// constraint row in the tableau, if none found return -1
		row = 2;
		while (true) {
			if (tableau[col][row] > Fractional(0)) {
				min_ratio = tableau["rhs"][row] / tableau[col][row];
				break;
			}

			row ++;

			if (row >= tableau[col].size())
				return -1;
		}

		// scan the rest of the constraints to see if there is a smaller ratio
		for (int i=row + 1; i < tableau[col].size(); ++i) {
			if(tableau[col][i] > Fractional(0)) {
				Fractional ratio = tableau["rhs"][i] / tableau[col][i];
				if(min_ratio > ratio) {
					min_ratio = ratio;
					row = i;
				}
			}
		}

		// return the row index of the minimum ratio for the given column
		return row;
	}

	bool pivot(string col, int row) {
		Fractional f(0);

		// divide the row through by the inverse of col,row
		f = tableau[col][row];
		for (vector<string>::iterator j=variables.begin();
				j != variables.end(); ++j)
			tableau[*j][row] = tableau[*j][row] / f;

		// zero out the column col by subtracting the value of row multiplied
		// by the negatino of col
		for (int i=0; i < rows; ++i) {
			f = tableau[col][i]; // the negation of col, for this row

			if (i == row)
				continue;

			for (vector<string>::iterator j=variables.begin();
					j != variables.end(); ++j) {
				tableau[*j][i] = tableau[*j][i] - (tableau[*j][row] * f);
			}
		}
	}

	string buildVariable(string prefix, int col) {
		char buffer[10];
		sprintf(buffer, "%i", col);
		return prefix + buffer;
	}

	FracVector buildIdentityCol(int row, Fractional coefficient) {
		int i;
		FracVector v;

		// loop through the rows that should be in the vector putting
		// coefficient in the correct row
		for(i=0; i < rows; ++i) {
			if(i == row)
				v.push_back(coefficient);
			else
				v.push_back(Fractional(0));
		}

		return v;
	}

	int addArtifical(string col, int row) {
		FracVector v = buildIdentityCol(row,Fractional(1));

		// add the row to the tableau
		v[0] = Fractional(1); // set Big-M coefficient
		tableau[col] = v;

		// do a pivot on the artifical col to put it in basis
		pivot(col,row);

		// put the value in basis for this row
		basic[row] = col;
	}

	bool loadTableau(FracVector &objective_function, vector< FracVector > &constraints) {
		map<string,FracVector>::iterator j;
		vector<int> artifical;
		int col, k, row;
		string s;

		// init the cols and rows for the class
		cols = 1;
		rows = 2 + constraints.size();  // always assume a Big-M row

		// build the objective function columns (including Big-M row)
		variables.push_back("z"); // z column always first

		tableau["z"].push_back(Fractional(0)); // Big-M row z column
		tableau["z"].push_back(Fractional(1)); // objective row z column

		for (FracVector::iterator i=objective_function.begin();
				i != objective_function.end(); ++i) {
			variables.push_back(s = buildVariable("x", cols));

			tableau[s].push_back(Fractional(0));
			tableau[s].push_back(Fractional((*i) * -1));

			cols++;
		}

		// add the Big-M and z row rhs value
		tableau["rhs"].push_back(Fractional(0));
		tableau["rhs"].push_back(Fractional(0));

		// make M the basic variable in row 0 and z basic in row 1
		basic[0] = "M";
		basic[1] = "z";

		// build the constraint rows (including extra variables)
		row = 2;
		for (vector< FracVector >::iterator i=constraints.begin();
				i != constraints.end(); ++i) {
			tableau[variables[0]].push_back(Fractional(0)); // push the z col

			for(col=0; col < (*i).size() - 2; ++col) {
				s = buildVariable("x", col+1);
				tableau[s].push_back((*i)[col]);
			}

			// add extra variables, push any artificals for process later
			switch ((*i)[(*i).size() - 2].getInt()) {
				case LESSER_EQUAL: { // add a slack variable
					variables.push_back(s = buildVariable("s", row - 1));
					tableau[s] = buildIdentityCol(row, Fractional(1));
					basic[row] = s;
					cols++;
				} break;
				case GREATER_EQUAL: { // add an excess and artifical variable
					variables.push_back(s = buildVariable("e", row - 1));
					tableau[s] = buildIdentityCol(row, Fractional(-1));
					cols++;
				} // fall through
				case EQUAL: { // add an artifical variable
					artifical.push_back(row);
				} break;
			}

			// add the rhs value for this constraint
			tableau["rhs"].push_back((*i)[(*i).size() - 1]);

			// incriment the row count
			row++;
		}

		// output the original problem tableau
		ofile << "Original problem" << endl << endl;
		outputTableau();
		ofile << endl;

		// do the artifical variables
		for (vector<int>::iterator i=artifical.begin(); i != artifical.end(); ++i) {
			bigM = true; // make sure we output a Big-M row
			variables.push_back(s = buildVariable("a", *i));
			variables.push_back("rhs"); // add it temporarly

			ofile << "Add " << s << " and pivot" << endl << endl;


			// call the function to add the artifical
			addArtifical(s,(*i));
			cols++;

			// output the new tableau
			outputTableau();
			ofile << endl;

			variables.pop_back(); // remove "rhs" so it'll always float to the end
		}

		// push the rhs symbol onto the end of the variables stack
		variables.push_back("rhs");

		return true;
	}

	bool isFeasable() {
		// see if there are any artifical variables in our solution
		for(map<int,string>::iterator i=basic.begin(); i != basic.end(); ++i) {
			if ((*i).second.find("a") != string::npos) {
				if (tableau["rhs"][(*i).first] == Fractional(0))
					continue;

				ofile << endl << "Found artifical variable in basis" << endl;
				return false;
			}
		}

		// other tests for infeasability

		// no infeasable conditions were found so exit with true
		return true;
	}

	bool isUnbounded() {

		// check every column of the tableau except z and rhs
		for(vector<string>::iterator i=variables.begin(); i != variables.end(); ++i) {
			if (*i == "z" || *i == "rhs")
				continue;

			// see if the row is less then zero
			if ((tableau[*i][0] < Fractional(0) &&
					!(tableau[*i][0] > Fractional(0))) ||
					tableau[*i][1] < Fractional(0)) {
				int cnt = 0;

				// count the number of negative values in this row
				for(FracVector::iterator j=tableau[*i].begin() + 2; j != tableau[*i].end(); ++j) {
					if (*j < Fractional(0))
						++cnt;
				}

				// if the count == the number of constraints (size() - 2) then
				// there is an unbounded variable
				if (cnt == tableau[*i].size() - 2)
					return true;
			}
		}

		// current solution is not unbounded
		return false;
	}

	bool solve() {
		string entering;
		int exiting;
		map<string,int> temp;

		// loop till we can't find another basic variable
		while ((entering = getEnteringBasic()) != "") {
			exiting = getExitingBasic(entering);

			if (exiting == -1)
				break;

			// pivot on the correct cell
			pivot(entering,exiting);

			// do some output and update the basis
			ofile << endl << entering << " enters the basis and "
				<< basic[exiting] << " leaves" << endl << endl;
			basic[exiting] = entering;
			outputTableau();
		}

		// check to see if we arrived at a feasable state
		if (!isFeasable()) {
			ofile << "Solution found is infeasable" << endl;

			return false;
		}

		ofile << endl << "Solution is optimal";

		// check for unboundedness
		if (isUnbounded()) {
			ofile << " and unbounded";
		}

		ofile << endl;

		// output the result
		ofile << "z = " << setw(8) << setprecision(2) << setiosflags(ios::fixed)
			<< setiosflags(ios::right) << tableau["rhs"][1].getFloat() << endl;
		ofile << "x = (";
		for(map<int,string>::iterator i=basic.begin(); i != basic.end(); ++i) {
			temp[(*i).second] = (*i).first;
		}

		for(vector<string>::iterator i=variables.begin(); i != variables.end(); ++i) {
			if((*i)[0] != 'x')
				continue;

			map<string,int>::iterator j = temp.find(*i);

			Fractional f = Fractional(0);
			if(j != temp.end())
				f = tableau["rhs"][(*j).second];

			ofile << setw(8) << setprecision(2) << setiosflags(ios::fixed)
				<< setiosflags(ios::right) << f.getFloat() << ", ";
		}
		ofile << ")" << endl;

		return true;
	}

	void outputTableau() {
		int row;

		// output a header row of the variables
		for(vector<string>::iterator i=variables.begin(); i != variables.end(); ++i)
			ofile << setw(8) << setiosflags(ios::right) << (*i);
		ofile << setw(8) << setiosflags(ios::right) << "basis";
		ofile << endl;

		// output a break line
		for(int i=0; i < variables.size() + 1; ++i) {
			ofile << "--------";
		}
		ofile << endl;

		// output the actual tableau
		for(row = 0; row < tableau["z"].size(); ++row) {
			if (!bigM && row == 0)
				continue;
			for(vector<string>::iterator i=variables.begin();
					i != variables.end(); ++i)
				ofile << setw(8) << setprecision(2) << setiosflags(ios::fixed)
					<< setiosflags(ios::right) << tableau[(*i)][row].getFloat();
			ofile << setw(8) << setiosflags(ios::right) << basic[row];
			ofile << endl;
		}
	}

private:
	map<string, FracVector> tableau;
	vector<string> variables;
	map<int,string> basic;
	int cols, rows;
	bool bigM;

	ofstream ofile;
};

#endif
