#ifndef __UTIL_H__
#define __UTIL_H__

#ifndef DEBUG
#define DEBUG 0
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
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

__host__ void matrix_allocate(matrix_t* m, int rows, int cols) {
	m->cols  = cols;
	m->rows  = rows;
	m->cells = new double*[rows];
	memset(m->cells,0,sizeof(double*) * rows);

	for(int i=0; i < rows; i++) {
		m->cells[i] = new double[cols];
		memset(m->cells[i],0,sizeof(double) * cols);
	}
}

__host__ void matrix_teardown(matrix_t &m) {
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

__host__ bool matrix_transpose(matrix_t* a, matrix_t* at) {
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

__host__ bool matrix_multiply(matrix_t* a, matrix_t* b, matrix_t* c) {
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

bool handleError(const char *error) {
	fprintf(stderr,"%s",error);
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

double compute_secs(const timespec *t) {
	return t->tv_sec + (t->tv_nsec / 1000000000.0);
}

void print_time(const timespec *t, const char *s) {
	double computed_time = compute_secs(t);
	printf("%s: %f sec\n", s, computed_time);
}

typedef struct {
	timespec start;
	timespec end;
	timespec elapsed;
} hr_timer_t;

bool timer_start(hr_timer_t* timer) {
//	clockid_t cid;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &timer->start);

/*	if (clock_getcpuclockid(0,&cid) == ENOENT) {
		debug("Process switched cpu which invalidated timer!");
		return false;
	}
*/
	return true;
}

bool timer_end(hr_timer_t* timer) {
//	clockid_t cid;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &timer->end);

/*	if (clock_getcpuclockid(0,&cid) == ENOENT) {
		debug("Process switched cpu which invalidated timer!");
		return false;
	}
*/
	time_diff(&timer->start,&timer->end,&timer->elapsed);

	return true;
}

#endif
