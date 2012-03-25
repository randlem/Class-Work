/*******************************************************************************
* Simplex Solver
*
* Written by: Mark Randles
*
* Usage: simplex <commands>
*	<commands>:
*		-f [filename]
*		-file [filename]
*		-o [filename]
*		-output [filename]
*******************************************************************************/

#include <iostream>
using std::cout;
using std::cin;
using std::cerr;
using std::endl;

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <map>
using std::map;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "simplex.h"
#include "fractional.h"
#include "options.h"

Options options;

bool init(Simplex*);

int main(int argc, char* argv[]) {
	Simplex* simplex;

	// setup the command line options
	options.setFlags("f","file");
	options.setFlags("file","file");
	options.setFlags("o","output");
	options.setFlags("output","output");

	// set some default options
	options.setOption("output","result.txt");

	// parse the command line
	options.parseCmdLine(argc,argv);

	// if we didn't get a file name on the command line, then prompt the user
	if (options.getOption("file") == "") {
		string t = "";
		cout << "Filename: ";
		cin >> t;
		options.setOption("file",t);
		cout << endl;
	}

	// create the new simplex object
	simplex = new Simplex(options.getOption("output"));

	// setup the simplex object
	init(simplex);

	// solve the problem
	simplex->solve();

	// cleanup memory
	delete simplex;

	return 0;
}

bool init(Simplex* simplex) {
	ifstream input;
	int m, n, cnt;
	string fbuffer;
	FracVector v;
	vector< FracVector > constraints;
	FracVector objective_function;

	// open the input file and make sure that we actually did it
	input.open(options.getOption("file").c_str());

	if(!input) {
		cerr << "ERROR: Couldn't open input file " <<
			options.getOption("file") << " for reading." << endl;
		return false;
	}

	// get the number of constraints and variables
	input >> m >> n;

	// make sure we've still got data to read
	if (input.eof()) {
		cerr << "ERROR: Invalid input format." << endl;
		return false;
	}

	// get the objective function
	for (int z=0; z < n; ++z) {
		input >> fbuffer;
		objective_function.push_back(Fractional(fbuffer));
	}

	// calc the number of coefficients for error checking
	cnt = m * (n + 2);

	// get the coefficients
	for (int z=0; z < m; ++z) {
		v.clear();

		for (int y=0; y < n + 2; ++y) {
			if (input.eof()) {
				cerr << "ERROR: Invalid input file." << endl;
				return false;
			}

			input >> fbuffer;
			v.push_back(Fractional(fbuffer));
			cnt--;
		}

		constraints.push_back(v);
	}

	// make sure we got the correct number of input variables
	if (cnt != 0) {
		cerr << cnt << "ERROR: Invalid input file." << endl;
		return false;
	}

	// close the input file
	input.close();

	cout << "The problem as read from input: " << endl;

	// output the objective function
	cout << "Max z = ";
	for (int z=0; z < n; ++z)
		cout << setw(8) << objective_function[z] << "*x" << z + 1 << ((z != n-1) ? " + " : "");
	cout << endl;

	// output the original constraints
	cout << "Constraints:" << endl;
	for (int z=0; z < m; ++z) {
		for (int y=0; y < n; ++y) {
			cout << setw(8) << constraints[z][y] << "*x" << y + 1 << ((y != n-1) ? " + " : "");
		}

		cout << " " << setw(3);
		switch (constraints[z][n].getInt()) {
			case LESSER_EQUAL: {
				cout << "<= ";
			} break;
			case GREATER_EQUAL: {
				cout << ">= ";
			} break;
			case EQUAL: {
				cout << "= ";
			} break;
		}

		cout << constraints[z][n+1] << endl;
	}
	cout << endl;

	// load the data into the simplex class
	simplex->loadTableau(objective_function,constraints);

	return true;
}
