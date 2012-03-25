#ifndef __TABLEAU_H__
#define __TABLEAU_H__

#include <iostream>
using std::cout;

#include <ostream>
using std::ostream;
using std::endl;

#include <vector>
using std::vector;

#include "fractional.h"

typedef vector<Fractional> RowType;

class Tableau {
	friend ostream& operator<<(ostream& output, Tableau& t) {
		vector< RowType >::iterator i;
		RowType::iterator j;

		for(i=t.tableau.begin(); i != t.tableau.end(); ++i) {
			for(j=(*i).begin(); j != (*i).end(); ++j) {
				output << (*j) << ((j + 1 != (*i).end()) ? " " : "");
			}
			output << endl;
		}

		return output;
	}

public:
	Tableau() {

	}

	~Tableau() {

	}

	bool scalarRow(int row, Fractional scalar) {
		RowType::iterator i;

		for(i=tableau[row].begin(); i != tableau[row].end(); ++i)
			(*i) = (*i) * scalar;

		return true;
	}

	bool scalarRow(RowType *v, Fractional scalar) {
		RowType::iterator i;

		for(i=v->begin(); i != v->end(); ++i)
			(*i) = (*i) * scalar;

		return true;
	}

	bool addRows(int row0, int row1) {
		int i;
		RowType::iterator pos0, pos1;

		for(i=0; i < tableau[row0].size(); ++i) {
			pos0 = tableau[row0].begin() + i;
			pos1 = tableau[row1].begin() + i;
			(*pos0) = (*pos0) + (*pos1);
		}

		return true;
	}

	bool addRows(int row0, RowType &row1) {
		int i;
		RowType::iterator pos0, pos1;

		for(i=0; i < tableau[row0].size(); ++i) {
			pos0 = tableau[row0].begin() + i;
			pos1 = row1.begin() + i;
			(*pos0) = (*pos0) + (*pos1);
		}

		return true;

	}

	bool pivot(int row, int col) {
		int i;

		// set the pivot element to zero by dividing the row by it's inverse
		scalarRow(row, tableau[row][col].inverse());

		// for each row which does not contain the pivot element zero out the pivot column
		for(i=0; i < tableau.size(); ++i) {
			if (i == row)
				continue;

			RowType v = tableau[row];
			scalarRow(&v, tableau[i][col] * -1);
			addRows(i, v);
		}

		return true;
	}

	int nextBasic() {
		int col = -1,
			cnt,
			i;
		Fractional max(0);

		// see if we've got a big M value and find the largest
		for (i=0; i < tableau[0].size(); ++i) {
			if(tableau[0][i] > Fractional(0) && max > tableau[0][i]) {
				max = tableau[0][i];
				col = i;
			}
		}

		// return a basic column if we found a big M value
		if (col != -1)
			return col;

		// find the smallest column coefficient in the z row
		for (i=0; i < tableau[1].size(); ++i) {
			if(max > tableau[1][i]) {
				max = tableau[1][i];
				col = i;
			}
		}

		// return the col value (-1 if none found)
		return col;
	}

	int pivotRow(int col) {
		int row = -1, i;
		int rhs = tableau[0].size() - 1;
		Fractional min(0);

		// loop through all of the rows and find the minimum ratio for a given
		// column
		min = tableau[2][rhs] / tableau[2][col];
		row = 2;
		for (i=2; i < tableau.size(); ++i) {
			// generate the ratio for this row
			Fractional ratio = tableau[i][rhs] / tableau[i][col];

			// see if this is a smaller ratio, save it if true
			if(min > ratio) {
				min = ratio;
				row = i;
			}
		}

		// return the row found (-1 if none)
		return row;
	}

	bool insertRow(int pos, RowType &row) {
		// check the size, fix to the end if we're too long
		if (pos >= tableau.size())
			pos = tableau.size();

		// insert the row
		tableau.insert(tableau.begin() + pos - 1, row);

		return true;
	}

	bool insertRow(RowType &row) {
		tableau.push_back(row);

		return true;
	}

	bool insertColumn(int pos, RowType &col) {
		int i;
		vector< RowType >::iterator j;

		// check to see if the tableau is empty
		if (tableau.empty()) {
			// the tableau is empty so we're going to be starting the row with
			// the new column so just do it this way
			for (i=0; i < col.size(); ++i) {
				RowType v;

				// add the new value to the vector
				v.push_back(col[i]);

				// insert the new row into the tableau
				tableau.insert(tableau.begin() + i, v);
			}

		} else {
			// insert each new column element into the correct row inserting
			// zeros when we run out of elements to add
			i = 0;
			for(j=tableau.begin(); j != tableau.end(); ++j) {
				(*j).insert((*j).begin() + pos - 1, (i >= col.size()) ? 0 : col[i++]);
			}
		}

		return true;
	}

	bool insertColumn(RowType &col) {
		int i;
		vector< RowType >::iterator j;

		// check to see if the tableau is empty
		if (tableau.empty()) {
			// the tableau is empty so we're going to be starting the row with
			// the new column so just do it this way
			for (i=0; i < col.size(); ++i) {
				RowType v;

				// add the new value to the vector
				v.push_back(col[i]);

				// insert the new row into the tableau
				tableau.insert(tableau.begin() + i, v);
			}

		} else {
			// insert each new column element into the correct row inserting
			// zeros when we run out of elements to add
			i = 0;
			for(j=tableau.begin(); j != tableau.end(); ++j)
				(*j).push_back((i >= col.size()) ? 0 : col[i++]);
		}

		return true;
	}

protected:
	vector< RowType > tableau;
};

#endif
