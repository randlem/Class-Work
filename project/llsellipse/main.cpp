#define DEBUG		 0 
#define FP_PRE        0.0000001
#define USAGE_MESSAGE "Usage: llsellipse <file>"
#define DATA_POINTS   6
#define EVIDENCE_TRSH 5018
#define F 100

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "util.h"
#include "gfxutil.h"
#include "imgutil.h"

typedef struct {
	int x;
	int y;
} edge_pixel_t, point_2d_t;

typedef struct {
	matrix_t center;
	matrix_t radius;
	int*     theta;
	int      theta_cnt;
} accumulator_t;

typedef struct {
	double x;
	double y;
	double a;
	double b;
	double theta;
} ellipse_t;

typedef struct {
	point_2d_t p1;
	point_2d_t p2;
	point_2d_t p3;
	point_2d_t p4;
} bezier_t;

string        in_filename        ;
string        out_filename       ;
img_t         in_image           ;
img_t         out_image          ;
edge_pixel_t* edge_pxls    = NULL;
int           edge_pxl_cnt = 0   ;
accumulator_t accum        = {{0, 0, NULL}, {0, 0, NULL}, NULL, 0};
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
void fit_points_5d(int, int*, double &, double &, double &, double &, double &);
void process_accum(ellipse_t*);
void find_max_2d(matrix_t*, int &, int &);
void gauss(matrix_t*, double*);

void rotate_point(point_2d_t *, point_2d_t &, double);
void plot_point(img_t*, uint_t, uint_t, const color_t &);
void bezier_curve(img_t*, bezier_t &, const color_t &);
void bezier_ellipse(img_t*, uint_t, uint_t, uint_t, uint_t, uint_t, const color_t &);
void bresenham_line(img_t*, point_2d_t , point_2d_t , const color_t &);

int main (int argc, char *argv[]) {
	ellipse_t ellipse;
	timespec start, end, t_ge, t_pa;

	// setup the program
	setup(&argc,argv);

	// gather evidence (timed)
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
	gather_evidence();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
	time_diff(&start,&end,&t_ge);
	//print_time(&diff,"gather_evidence");

	// recover the circle from the evidence
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
	process_accum(&ellipse);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
	time_diff(&start,&end,&t_pa);
	//print_time(&diff,"process_accum");
	debug("recovered ellipse:\n\tcentered = (%0.2f,%0.2f)\n\tsemi-axis = (%0.2f,%0.2f)\n\ttheta = %0.2f",ellipse.x,ellipse.y,ellipse.a,ellipse.b,ellipse.theta);

	cout << (t_ge.tv_sec + (t_ge.tv_nsec / 1000000000.0)) << " "
	     << (t_pa.tv_sec + (t_pa.tv_nsec / 1000000000.0)) << " "
		 << (t_ge.tv_sec + (t_ge.tv_nsec / 1000000000.0)) + (t_pa.tv_sec + (t_pa.tv_nsec / 1000000000.0)) << " "
		 << ellipse.x << " " << ellipse.y << " " << ellipse.a << " "
		 << ellipse.b << " " << ellipse.theta << endl;

	// cleanup the program
//	terminate();
	return 0;
}

void setup(int* argc, char **argv) {
	int pos;

	// check the lenght of the cmd line
	if (*argc != 2)
		cerr << USAGE_MESSAGE << endl;

	// seed the rng
	srand(clock());

	// load the input image
	debug("Reading in file %s...",argv[1]);
	in_filename = argv[1];
	image_read(argv[1],&in_image);

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

	// allocate the accumulator for the ellipse center
	matrix_allocate(&accum.radius,in_image.width,in_image.height);

	// allocate the accumulator for the theta
	accum.theta_cnt = 180;
	accum.theta     = new int[accum.theta_cnt];
	memset(accum.theta,0,sizeof(int) * accum.theta_cnt);
}

void setup_lls() {
	matrix_allocate(&aug, 5          , 6);
	matrix_allocate(&X  , DATA_POINTS, 5);
	matrix_allocate(&Xt , 5          , DATA_POINTS);
	matrix_allocate(&XtX, 5          , 5);
	matrix_allocate(&y  , DATA_POINTS, 1);
	matrix_allocate(&Xty, 5          , 1);
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
	delete [] accum.theta;
	matrix_teardown(accum.radius);
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
	debug("edge pixel cnt = %d",edge_pxl_cnt);

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
	int evidence, *pxls;
	double A,B,C,D,E,j,delta,r1,r2,t,slope1,slope2;
	ellipse_t ellipse;

	// allocate the pixels storage
	pxls = new int[DATA_POINTS];

	evidence = 0;
	while (evidence < EVIDENCE_TRSH) {
		memset(&ellipse,0,sizeof(ellipse_t));

		// randomly select DATA_POINTS # of edge pixels
		for (int i=0; i < DATA_POINTS; i++)
			pxls[i] = random_int(0,edge_pxl_cnt);

		// fit the points we selected to the general quadratic
		fit_points_5d(DATA_POINTS,pxls,A,B,C,D,E);

		// calc j to determine if we have an ellipse
		j = (A * B) - ((C * C) / 4.0);

		// determine if we have a circle
		if (j > 0.0 && fabs(j) > FP_PRE) {
			// recover the parameters
			delta = (A * B * -F)     +
					(C * E * D)  / 8 +
					(D * C * E)  / 8 -
					(D * D * B)  / 4 -
					(A * E * E)  / 4 -
					(C * C * -F) / 4;
			t     = sqrt((B - A) * (B - A) + C * C);
			r1    = (A + B + t) / 2.0;
			r2    = (A + B - t) / 2.0;

			ellipse.x     = (2.0 * B * D - C * E) / (C * C - 4.0 * A * B);
			ellipse.y     = (2.0 * A * E - C * D) / (C * C - 4.0 * A * B);
			ellipse.a     = sqrt(fabs(delta) / fabs(j * r2));
			ellipse.b     = sqrt(fabs(delta) / fabs(j * r1));
			t             = (B - A) / C;
			slope1        = sqrt((t * t) + 1.0) + t;
			slope2        = -sqrt((t * t) + 1.0) + t;
			ellipse.theta = atan(slope2) / M_PI * 180;

			if (ellipse.b > ellipse.a) {
				t = ellipse.b;
				ellipse.b = ellipse.a;
				ellipse.a = t;
				ellipse.theta = atan(slope1) / M_PI * 180 + 0.5;

				if (C < 0)
					ellipse.theta += 90;
			}

			// put them in the accumulator
			if (in_range((int)ellipse.theta,-90,90) &&
				in_range((int)ellipse.x,0,accum.center.cols-1) && in_range((int)ellipse.y,0,accum.center.rows-1) &&
				in_range((int)ellipse.a,0,accum.center.cols-1) && in_range((int)ellipse.b,0,accum.center.rows-1)) {
				accum.center.cells[(int)ellipse.y][(int)ellipse.x]++;
				accum.radius.cells[(int)ellipse.b][(int)ellipse.a]++;
				accum.theta[(int)ellipse.theta + 90]++;
			}
		}
		evidence++;
	}
}

// fit a set of points to the general quadratic:
//  Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 where F == -1
void fit_points_5d(int pxls_cnt, int* pxls, double &A, double &B, double &C, double &D, double &E) {
	int i,j;
	double *results;

	// allocate
	results = new double[5];
	memset(results,0,sizeof(double) * 5);

	// fill the X matrix
	for (i=0; i < pxls_cnt; i++) {
		X.cells[i][0] = edge_pxls[pxls[i]].x * edge_pxls[pxls[i]].x;
		X.cells[i][1] = edge_pxls[pxls[i]].y * edge_pxls[pxls[i]].y;
		X.cells[i][2] = edge_pxls[pxls[i]].x * edge_pxls[pxls[i]].y;
		X.cells[i][3] = edge_pxls[pxls[i]].x;
		X.cells[i][4] = edge_pxls[pxls[i]].y;
	}

	// fill the Y matrix
	for (i=0; i < pxls_cnt; i++) {
		y.cells[i][0] = F;
	}

	// create the X transpose matrix
	matrix_transpose(&X,&Xt);

//	matrix_print("X",&X);
//	matrix_print("Xt",&Xt);

	// multiply X and Xt into the augmented matrix
	matrix_multiply(&Xt,&X,&XtX);

//	matrix_print("XtX",&XtX);

	// multiply y by the transpose of X
	matrix_multiply(&Xt,&y,&Xty);

	// copy the results into the augmented matrix
	for (i=0; i < Xty.rows; i++) {
		for(j=0; j < XtX.cols; j++)
			aug.cells[i][j] = XtX.cells[i][j];
		aug.cells[i][5] = Xty.cells[i][0];
	}

	// run a gaussian solver on the system of equations
//	matrix_print("Augmented (pre-gauss)",&aug);
	gauss(&aug,results);
//	matrix_print("Augmented (post-gauss)",&aug);

	// extract the results
	A = results[0];
	B = results[1];
	C = results[2];
	D = results[3];
	E = results[4];

	// cleanup
	delete [] results;

}

void process_accum(ellipse_t *e) {
	int i,j,max_i,max_j;

	max_i = 0;
	for (i=1; i < accum.theta_cnt; i++) {
		if (accum.theta[max_i] < accum.theta[i])
			max_i = i;
	}
	e->theta = max_i - 90;

	find_max_2d(&accum.center,max_i,max_j);
	e->y = max_i;
	e->x = max_j;

	find_max_2d(&accum.radius,max_i,max_j);
	e->b = max_i;
	e->a = max_j;
}

void find_max_2d(matrix_t* m, int &max_i, int &max_j) {
	int i,j;

	max_i = 0;
	max_j = 0;
	for (i=0; i < m->rows; i++) {
		for (j=0; j < m->cols; j++) {
			if (m->cells[max_i][max_j] < m->cells[i][j]) {
				max_i = i;
				max_j = j;
			}
		}
	}
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

void rotate_point(point_2d_t *p, point_2d_t &about, double angle) {
	double theta;
	int x = p->x - about.x,y = p->y - about.y;

	while (angle > 360.0)
		angle -= 360.0;

	while (angle < 0.0)
		angle += 360.0;

	theta = (float)(angle * ((2 * M_PI) / 360.0));
	p->x = ((int)round((double)x * cos(theta) - (double)y * sin(theta))) + about.x;
	p->y = ((int)round((double)x * sin(theta) + (double)y * cos(theta))) + about.y;
}

void plot_point(img_t* img, uint_t x, uint_t y, const color_t &c) {
	double draw = 0.0;
	int offset  = 0;

	if (y > img->height-1)
		return;
	if (x > img->width-1)
		return;

	img->pixels[y][x].rgba = c.rgba;
}

void bezier_curve(img_t* img, bezier_t &b, const color_t &c) {
	float t, t3, t2, c0, c1, c2, c3, pitch;
	int deltax, deltay;
	point_2d_t p1,p2;

	deltax = abs(b.p1.x - b.p4.x);
	deltay = abs(b.p1.y - b.p4.y);
	pitch  = 3.0 / ((deltax > deltay) ? deltax : deltay);

	p1.x = b.p1.x;
	p1.y = b.p1.y;
	for (t=pitch; t < 1.0; t+=pitch) {
		c0 = (1 - t) * (1 - t) * (1 - t);
		c1 = 3 * ((1 - t) * (1 - t)) * t;
		c2 = 3 * (1 - t) * (t * t);
		c3 = t * t * t;

		p2.x = (int)round(c0 * b.p1.x + c1 * b.p2.x + c2 * b.p3.x + c3 * b.p4.x);
		p2.y = (int)round(c0 * b.p1.y + c1 * b.p2.y + c2 * b.p3.y + c3 * b.p4.y);

		bresenham_line(img,p1,p2,c);

		p1.x = p2.x;
		p1.y = p2.y;
	}
	bresenham_line(img,p1,b.p4,c);
}

void bezier_ellipse(img_t* img, uint_t a, uint_t b, uint_t rx, uint_t ry, uint_t angle, const color_t &c) {
	bezier_t b1,b2,b3,b4;
	uint_t x0,x1,y0,y1,rxo,ryo;
	point_2d_t center;

	center.x = a;
	center.y = b;

	x0 = a - rx;
	x1 = a + rx;
	y0 = b - ry;
	y1 = b + ry;
	rxo = (int)(2 * rx * 0.2761423749154);
	ryo = (int)(2 * ry * 0.2761423749154);

	// top-left quadrant
	b1.p1.x = x0;
	b1.p1.y = b;
	b1.p2.x = x0;
	b1.p2.y = b + ryo;
	b1.p3.x = a - rxo;
	b1.p3.y = y1;
	b1.p4.x = a;
	b1.p4.y = y1;
	rotate_point(&b1.p1,center,angle);
	rotate_point(&b1.p2,center,angle);
	rotate_point(&b1.p3,center,angle);
	rotate_point(&b1.p4,center,angle);

	// top-right quadrant
	b2.p1.x = b1.p4.x;
	b2.p1.y = b1.p4.y;
	b2.p2.x = a + rxo;
	b2.p2.y = y1;
	b2.p3.x = x1;
	b2.p3.y = b + ryo;
	b2.p4.x = x1;
	b2.p4.y = b;
	rotate_point(&b2.p2,center,angle);
	rotate_point(&b2.p3,center,angle);
	rotate_point(&b2.p4,center,angle);

	// lower-right quadrant
	b3.p1.x = b2.p4.x;
	b3.p1.y = b2.p4.y;
	b3.p2.x = x1;
	b3.p2.y = b - ryo;
	b3.p3.x = a + rxo;
	b3.p3.y = y0;
	b3.p4.x = a;
	b3.p4.y = y0;
	rotate_point(&b3.p2,center,angle);
	rotate_point(&b3.p3,center,angle);
	rotate_point(&b3.p4,center,angle);

	// lower-left quadrant
	b4.p1.x = b3.p4.x;
	b4.p1.y = b3.p4.y;
	b4.p2.x = a - rxo;
	b4.p2.y = y0;
	b4.p3.x = x0;
	b4.p3.y = b - ryo;
	b4.p4.x = b1.p1.x;
	b4.p4.y = b1.p1.y;
	rotate_point(&b4.p2,center,angle);
	rotate_point(&b4.p3,center,angle);

	bezier_curve(img,b1,c);
	bezier_curve(img,b2,c);
	bezier_curve(img,b3,c);
	bezier_curve(img,b4,c);
}

void bresenham_line(img_t* img, point_2d_t p1, point_2d_t p2, const color_t &c) {
	int deltax, deltay,
		dfa, dfb, error,
		stepx, stepy,
		x, y, t;
	point_2d_t p;
	bool steep;

	// precompute some values
	deltax = (p2.x - p1.x);
	deltay = (p2.y - p1.y);

	// figure out if this is a "steep" line.
	steep = (abs(deltay) > abs(deltax));
	if (steep) {
		t = p1.y; p1.y = p1.x; p1.x = t;
		t = p2.y; p2.y = p2.x; p2.x = t;
	}

	// put the small x on the left.
	if (p1.x > p2.x) {
		t = p2.x; p2.x = p1.x; p1.x = t;
		t = p2.y; p2.y = p1.y; p1.y = t;
	}

	// calc the x/y deltas
	deltax = p2.x - p1.x;
	deltay = abs(p2.y - p1.y);

	// calc the error
	error = deltax / 2;

	// setup the stepping vals.
	stepx = 1;
	stepy = (p1.y < p2.y) ? 1 : -1;

	// draw the line
	y = p1.y;
	for (x=p1.x; x <= p2.x; x+=stepx) {
		if (steep) {
			p.x = y;
			p.y = x;
		} else {
			p.x = x;
			p.y = y;
		}

		plot_point(img, p.x, p.y, c);

		error -= deltay;
		if (error < 0) {
			error += deltax;
			y += stepy;
		}
	}
}
