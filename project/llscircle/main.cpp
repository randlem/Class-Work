#define DEBUG		  1
#define FP_PRE        0.0000001
#define USAGE_MESSAGE "Usage: llscircle <file>"
#define DATA_POINTS   4
#define EVIDENCE_TRSH 10000

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <time.h>
#include <math.h>

#include "util.h"
#include "imgutil.h"
#include "gfxutil.h"

typedef struct {
	int x;
	int y;
} edge_pixel_t;

typedef struct {
	matrix_t center;
	int*  radius;
	int   radius_cnt;
} accumulator_t;

typedef struct {
	double a;
	double b;
	double r;
} circle_t;

string        in_filename        ;
string        out_filename       ;
img_t         in_image           ;
img_t         out_image          ;
edge_pixel_t* edge_pxls    = NULL;
int           edge_pxl_cnt = 0   ;
accumulator_t accum        = {{0, 0, NULL}, NULL, 0};
matrix_t      X            = {0, 0, NULL};
matrix_t      Xt           = {0, 0, NULL};
matrix_t      XtX          = {0, 0, NULL};
matrix_t      y            = {0, 0, NULL};
matrix_t      Xty          = {0, 0, NULL};
matrix_t      aug          = {0, 0, NULL};

void setup(int*, char**);
void setup_accum();
void setup_lls();

void terminate();
void shutdown_accum();
void shutdown_lls();

void process_input();
void gather_evidence();
edge_pixel_t* random_points(int);
void fit_points_3d(int, int*, double &, double &, double &);
void process_accum(circle_t*);
void gauss(matrix_t*, double*);

void plot_point(img_t*, uint_t, uint_t, const color_t &);
void bresenham_circle(img_t*, uint_t, uint_t, uint_t, const color_t &);

int main (int argc, char *argv[]) {
	circle_t circle;

	// setup the program
	setup(&argc,argv);

	// gather evidence
	gather_evidence();

	// recover the circle from the evidence
	process_accum(&circle);
	debug("recovered circle centered (%f,%f) radius %f",circle.a, circle.b, circle.r);

	// draw the detected circle on a copy of the original image
	image_copy(&in_image,&out_image);
	bresenham_circle(&out_image, (uint_t)circle.a, (uint_t)circle.b, (uint_t)circle.r, ORANGE);
	image_write(out_filename,&out_image);

	// cleanup the program
	//terminate();

	return 0;
}

void setup(int* argc, char **argv) {
	int pos;

	// check the lenght of the cmd line
	if (*argc != 2)
		cerr << USAGE_MESSAGE << endl;

	// seed the rng
	srand(time(NULL));

	// load the input image
	debug("Reading in file %s...",argv[1]);
	in_filename = argv[1];
	image_read(in_filename,&in_image);

	// create the output filename
	pos = in_filename.find_last_of(".");
	if (pos != string::npos)
		out_filename = in_filename.substr(0,pos) + string(".out.png");
	else
		out_filename = in_filename + string(".out.png");
	debug("Output filename: %s",out_filename.c_str());

	// process the input image into edge pixels
	process_input();

	// setup the accumulator
	setup_accum();

	// setup the matrix for the lls fitting
	setup_lls();
}

void setup_accum() {
	// allocate the accumulator for the ellipse center
	matrix_allocate(&accum.center,in_image.width,in_image.height);

	// allocate the accumulator for the theta
	accum.radius_cnt = in_image.height / 2;
	accum.radius     = new int[accum.radius_cnt];
	memset(accum.radius,0,sizeof(int) * accum.radius_cnt);
}

void setup_lls() {
	matrix_allocate(&aug, 3          , 4);
	matrix_allocate(&X  , DATA_POINTS, 3);
	matrix_allocate(&Xt , 3          , DATA_POINTS);
	matrix_allocate(&XtX, 3          , 3);
	matrix_allocate(&y  , DATA_POINTS, 1);
	matrix_allocate(&Xty, 3          , 1);
}

void terminate() {
	if (edge_pxls != NULL) {
		delete [] edge_pxls;
		edge_pxls    = NULL;
		edge_pxl_cnt = 0;
	}

	shutdown_accum();
	shutdown_lls();
}

void shutdown_accum() {
	delete [] accum.radius;
	matrix_teardown(accum.center);
}

void shutdown_lls() {
	matrix_teardown(Xty);
	matrix_teardown(y);
	matrix_teardown(XtX);
	matrix_teardown(Xt);
	matrix_teardown(X);
	matrix_teardown(aug);
}

void process_input() {
	int y, x, i;

	edge_pxl_cnt = 0;
	for (y=0; y < in_image.height; y++) {
		for(x=0; x < in_image.width; x++) {
			if (in_image.pixels[y][x].rgba != 0)
				edge_pxl_cnt++;
		}
	}

	edge_pxls = new edge_pixel_t[edge_pxl_cnt];
	memset(edge_pxls,0,sizeof(edge_pixel_t) * edge_pxl_cnt);

	i = 0;
	for (y=0; y < in_image.height; y++) {
		for(x=0; x < in_image.width; x++) {
			if (in_image.pixels[y][x].rgba != 0) {
				edge_pxls[i].x = x;
				edge_pxls[i].y = y;
				i++;
			}
		}
	}
}

void gather_evidence() {
	int good_evidence, attempted, *pxls;
	double A,B,C,D,E,F,j,delta,C2,t;
	circle_t circle;

	debug("Gathering evidence...");

	// allocate the pixels storage
	pxls = new int[DATA_POINTS];

	good_evidence = 0;
	attempted = 0;
	while (good_evidence < EVIDENCE_TRSH) {
		memset(&circle,0,sizeof(circle_t));

		// randomly select DATA_POINTS # of edge pixels
		for (int i=0; i < DATA_POINTS; i++)
			pxls[i] = random_int(0,edge_pxl_cnt);

		// fit the points we selected to the general quadratic
		fit_points_3d(DATA_POINTS,pxls,D,E,F);

		// determine if we have a circle
		if (F > 0) {
			// recover the parameters
			circle.a = D / -2.0;
			circle.b = E / -2.0;
			circle.r = sqrt(fabs(F - (circle.a * circle.a) - (circle.b * circle.b)));

			// put them in the accumulator
			circle.a = round(circle.a);
			circle.b = round(circle.b);
			circle.r = round(circle.r);
			if (circle.r < accum.radius_cnt && in_range((int)circle.a,0,accum.center.cols-1) && in_range((int)circle.b,0,accum.center.rows-1)) {
				accum.center.cells[(int)circle.b][(int)circle.a]++;
				accum.radius[(int)circle.r]++;

				// increment the evidence counter
				good_evidence++;
			}
		}
		attempted++;
	}
	debug("attempted = %d good = %d",attempted,good_evidence);

}

// fit a set of points to the general quadratic:
//  Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 where A=C=1 & B=0
void fit_points_3d(int pxls_cnt, int* pxls, double &D, double &E, double &F) {
	int i,j;
	double *results;

	// allocate
	results = new double[3];
	memset(results,0,sizeof(double) * 3);

	// fill the X matrix
	for (i=0; i < pxls_cnt; i++) {
		X.cells[i][0] = edge_pxls[pxls[i]].x;
		X.cells[i][1] = edge_pxls[pxls[i]].y;
		X.cells[i][2] = 1;
	}

	// fill the Y matrix
	for (i=0; i < pxls_cnt; i++) {
		y.cells[i][0] = -((edge_pxls[pxls[i]].x * edge_pxls[pxls[i]].x) +
			(edge_pxls[pxls[i]].y * edge_pxls[pxls[i]].y));
	}

	// create the X transpose matrix
	matrix_transpose(&X,&Xt);

	// multiply X and Xt into the augmented matrix
	matrix_multiply(&Xt,&X,&XtX);

	// multiply y by the transpose of X
	matrix_multiply(&Xt,&y,&Xty);

	// copy the results into the augmented matrix
	for (i=0; i < Xty.rows; i++) {
		for(j=0; j < XtX.cols; j++)
			aug.cells[i][j] = XtX.cells[i][j];
		aug.cells[i][XtX.cols] = Xty.cells[i][0];
	}

	// run a gaussian solver on the system of equations
	gauss(&aug,results);

	// extract the results
	D = results[0];
	E = results[1];
	F = results[2];

	// cleanup
	delete [] results;
}

// fit a set of points to the general quadratic:
//  Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
void fit_points_5d(int pxls_cnt, int* pxls, double &A, double &B, double &C, double &D, double &E) {
	int i,j;
	double *results;

	// allocate
	results = new double[5];
	memset(results,0,sizeof(double) * 5);

	// fill the X matrix
	for (i=0; i < pxls_cnt; i++) {
		X.cells[i][0] = edge_pxls[pxls[i]].x * edge_pxls[pxls[i]].x;
		X.cells[i][1] = edge_pxls[pxls[i]].x * edge_pxls[pxls[i]].y;
		X.cells[i][2] = edge_pxls[pxls[i]].y * edge_pxls[pxls[i]].y;
		X.cells[i][3] = edge_pxls[pxls[i]].x;
		X.cells[i][4] = edge_pxls[pxls[i]].y;
	}

	// fill the Y matrix
	for (i=0; i < pxls_cnt; i++) {
		y.cells[i][0] = -1;
	}

	// create the X transpose matrix
	matrix_transpose(&X,&Xt);

	// multiply X and Xt into the augmented matrix
	matrix_multiply(&Xt,&X,&XtX);

	// multiply y by the transpose of X
	matrix_multiply(&Xt,&y,&Xty);

	// copy the results into the augmented matrix
	for (i=0; i < Xty.rows; i++) {
		for(j=0; j < XtX.cols; j++)
			aug.cells[i][j] = XtX.cells[i][j];
		aug.cells[i][5] = Xty.cells[i][0];
	}

	// run a gaussian solver on the system of equations
	gauss(&aug,results);

	// extract the results
	A = results[0];
	B = results[1];
	C = results[2];
	D = results[3];
	E = results[4];

	// cleanup
	delete [] results;

}

void process_accum(circle_t *c) {
	int i,j,max_i,max_j;

	max_i = 0;
	for (i=1; i < accum.radius_cnt; i++) {
		if (accum.radius[max_i] < accum.radius[i])
			max_i = i;
	}
	c->r = max_i;

	max_i = 0;
	max_j = 0;
	for (i=0; i < accum.center.rows; i++) {
		for (j=0; j < accum.center.cols; j++) {
			if (accum.center.cells[max_i][max_j] < accum.center.cells[i][j]) {
				max_i = i;
				max_j = j;
			}
		}
	}

	c->b = max_i;
	c->a = max_j;
}

// implemented from psudocode at: http://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode
// gaussian elimination with partial pivoting
// results is ordered x1,x2,x3...xn
void gauss(matrix_t* A, double* results) {
	int i,j,k,u,maxi;
	double var, *tp;

	i = 0;
	j = 0;
	while (i < A->rows && j < A->cols) {
		// find the row with the maximum value
		maxi = i;
		for (k=i+1; k < A->rows; k++) {
			if (fabs(A->cells[k][j]) > fabs(A->cells[maxi][j]))
				maxi = k;
		}

		if (A->cells[maxi][j] != 0) {
			// swap rows
			 if (i != maxi) {
				tp = A->cells[i];
				A->cells[i] = A->cells[maxi];
				A->cells[maxi] = tp;
			}

			// reduce pivot element to 1
			var = A->cells[i][j];
			for (k=0; k < A->cols; k++)
				A->cells[i][k] /= var;

			// remove the pivot element from all subsequent rows
			for (u=i+1; u < A->rows; u++) {
				var = A->cells[u][j];
				for (k=j; k < A->cols; k++)
					A->cells[u][k] -= A->cells[i][k] * var;
			}

			i++;
		}

		j++;
	}

	// retrieve the results
	for (i=A->rows-1; i >= 0; i--) {
		var = A->cells[i][A->cols-1];
		for (j=A->cols-2; j > i; j--) {
			var -= A->cells[i][j] * results[j];
		}
		results[i] = var;
	}

}

void plot_point(img_t* img, uint_t x, uint_t y, const color_t &c) {
	if (y > img->height-1)
		return;
	if (x > img->width-1)
		return;

	img->pixels[y][x].rgba = c.rgba;
}

void bresenham_circle(img_t* img, uint_t a, uint_t b, uint_t r, const color_t &c) {
	int y    = r;
	int d    = -r;
	int x2m1 = -1;
	int x;

	plot_point(img, a, b + r, c);
	plot_point(img, a, b - r, c);
	plot_point(img, a + r, b, c);
	plot_point(img, a - r, b, c);
	for(x=1; x < r / sqrt(2); x++) {
		x2m1 += 2;
		d += x2m1;

		if (d >= 0) {
			y--;
			d -= (y<<1);
		}

		plot_point(img, a + x, b + y, c);
		plot_point(img, a + x, b - y, c);
		plot_point(img, a - x, b + y, c);
		plot_point(img, a - x, b - y, c);
		plot_point(img, a + y, b + x, c);
		plot_point(img, a + y, b - x, c);
		plot_point(img, a - y, b + x, c);
		plot_point(img, a - y, b - x, c);
	}
}
