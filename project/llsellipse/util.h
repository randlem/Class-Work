#ifndef __UTIL_H__
#define __UTIL_H__

#ifndef DEBUG
#define DEBUG 0
#endif

#include <iostream>
using std::cerr;
using std::endl;

#include <iomanip>
using std::ios;

#include <string>
using std::string;

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned char uchar_t;
typedef unsigned int  uint_t;

typedef struct {
	int      cols;
	int      rows;
	double** cells;
} matrix_t;

void debug(const char * s, ...) {
	if (DEBUG > 0) {
		va_list args;
		va_start(args, s);
		vfprintf(stdout, s, args);
		fprintf(stdout, "\n");
		va_end(args);
	}
}

bool in_range(int val, int low, int high) {
	return low <= val && val <= high;
}

int random_int(int low, int high) {
	return (rand() % high) + low;
}

void matrix_allocate(matrix_t* m, int rows, int cols) {
	m->cols  = cols;
	m->rows  = rows;
	m->cells = new double*[rows];
	memset(m->cells,0,sizeof(double*) * rows);

	for(int i=0; i < rows; i++) {
		m->cells[i] = new double[cols];
		memset(m->cells[i],0,sizeof(double) * cols);
	}
}

void matrix_teardown(matrix_t &m) {
	if (m.cells == NULL)
		return;

	for(int i=0; i < m.rows; i++) {
		if (m.cells[i] != NULL)
			delete [] m.cells[i];
	}
	delete [] m.cells;
	m.cells = NULL;
	m.rows  = 0;
	m.cols  = 0;
}

bool matrix_transpose(matrix_t* a, matrix_t* at) {
	if (a->cols != at->rows)
		return false;

	if (a->rows != at->cols)
		return false;

	for(int i=0; i < a->rows; i++) {
		for(int j=0; j < a->cols; j++) {
			at->cells[j][i] = a->cells[i][j];
		}
	}

	return true;
}

bool matrix_multiply(matrix_t* a, matrix_t* b, matrix_t* c) {
	double sum = 0.0;

	if (a->cols != b->rows)
		return false;

	//matrix_allocate(c,a->rows,b->cols);

	for (int i=0; i < a->rows; i++) {
		for (int j=0; j < b->cols; j++) {
			c->cells[i][j] = 0.0;
			for (int m=0; m < a->cols; m++) {
				c->cells[i][j] += a->cells[i][m] * b->cells[m][j];
			}
		}
	}

	return true;
}

void matrix_print(const string s, const matrix_t* a) {
	cout << s << " ("<< a->rows << "x" << a->cols << ")" << endl;
	for (int i=0; i < a->rows; i++) {
		for (int j=0; j < a->cols; j++) {
			cout.width(8);
			cout.fill(' ');
			if (a->cells[i][j] < 10.0)
				cout.precision(5);
			else if (a->cells[i][j] < 100.0)
				cout.precision(4);
			else if (a->cells[i][j] < 1000.0)
				cout.precision(3);
			else
				cout.precision(2);
			cout.flags(ios::right | ios::fixed);
			cout << a->cells[i][j];
			if ((j+1) != a->cols)
				cout << " ";
		}
		cout << endl;
	}
}

bool handleError(string message) {
	cerr << message << endl;
	return false;
}

void time_diff(timespec* start, timespec* end, timespec* diff) {
	if ((end->tv_nsec - start->tv_nsec) < 0) {
		diff->tv_sec  = end->tv_sec - start->tv_sec - 1;
		diff->tv_nsec = 1000000000 + end->tv_nsec - start->tv_nsec;
	} else {
		diff->tv_sec  = end->tv_sec - start->tv_sec;
		diff->tv_nsec = end->tv_nsec - start->tv_nsec;
	}
}

void print_time(const timespec *t, const char *s) {
	double computed_time = t->tv_sec + (t->tv_nsec / 1000000000.0);
	printf("%s: %f sec\n", s, computed_time);
}

#endif
