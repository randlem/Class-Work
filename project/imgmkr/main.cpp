/**********************************************************
* IMGMKR -- Image generation program
*
* Generates a PNG image from a text-based input file.  Also
* capable of drawing objects on a previously created file.
* Shapes able to be drawn are lines, circles, ellipses,
* and anything definable by a bezier curve.  Supports
* 32-bit color, no alpha channel.
*
* Debugging messages are controlled by the DEBUG define.
*     1 = Debug messages on
*     0 = Debug messages off
*
* Multiple types of noise can be specified: background,
*     shape noise, and perterb
*
*     background: uniformly distributed gaussian noise
*     shape noise: creates gaps in shapes
*     perterb: uses gaussian distribution to move pixels
*         in a circle
*
**********************************************************/

#define DEBUG		  1
#define USAGE_MESSAGE "Usage: imgmkr <file>"

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <fstream>
using std::ifstream;

#include <string>
using std::string;

#include <time.h>
#include <math.h>
#include "util.h"
#include "imgutil.h"

// stores a range of numbers bounded by two integers
typedef struct {
	int low;
	int high;
} range_t;

// stores a point in 2d space
typedef struct {
	uint_t x;
	uint_t y;
} point_2d_t;

// stores the parameters of a circle
typedef struct {
	uint_t a;
	uint_t b;
	uint_t r;
} circle_t;

// stores the parameters of an ellipse
typedef struct {
	uint_t a;
	uint_t b;
	uint_t rx;
	uint_t ry;
	uint_t angle;
} ellipse_t;

// stores the points of a bezier curve
typedef struct {
	point_2d_t p1;
	point_2d_t p2;
	point_2d_t p3;
	point_2d_t p4;
} bezier_t;

ifstream  in_file;				// text input file
uint	  edge_pxls_drawn = 0;	// global pixel counter

void setup(int*, char**);
void terminate();

void explode_range(string*, range_t*);

double random_uniform();
double random_uniform_range(range_t &);
double gauss_random(double mu, double sigma);

void rotate_point(point_2d_t *, point_2d_t &, double);

void flood_fill(img_t*, const color_t &);
void plot_point(img_t*, uint_t, uint_t, const color_t &, double, int);
void bezier_curve(img_t*, bezier_t &, const color_t &, double, int);
void bezier_ellipse(img_t*, uint_t, uint_t, uint_t, uint_t, uint_t, const color_t &, double, int);
void bresenham_line(img_t*, point_2d_t , point_2d_t , const color_t &, double, int);
void bresenham_circle(img_t*, uint_t, uint_t, uint_t, const color_t &, double, int);
void bresenham_ellipse(img_t*, uint_t, uint_t, uint_t, uint_t, uint_t, const color_t &, double, int);
void background_noise(img_t*, double, const color_t &);

int main (int argc, char *argv[]) {
	string out_filename,cmd,rng;
	int width, height, perterb, r, g, b;
	float noisy,bg_noise;
	color_t color;
	circle_t circle;
	ellipse_t ellipse;
	bezier_t bezier;
	point_2d_t point;
	range_t range;
	img_t img;

	// setup the program
	setup(&argc,argv);

	// clear out some useful structures
	memset(&color,0,sizeof(color_t));
	memset(&circle,0,sizeof(circle_t));
	memset(&ellipse,0,sizeof(ellipse_t));
	noisy   = 0;
	perterb = 0;

	// process the input file preamble
	in_file >> out_filename >> width >> height;

	// figure out if we're loading a previously generated image or not
	if (width == 0 && height == 0) {
		debug("Opening %s for reading...",out_filename.c_str());
		image_read(out_filename,&img);
		width = img.width;
		height = img.height;
	} else {
		debug("%s %dx%d",out_filename.c_str(), width, height);
		image_create(&img, width, height);
	}

	// MAIN LOOP
	in_file >> cmd;
	while (cmd != "end") {
		debug("%s",cmd.c_str());

		// process the input command
		if (cmd == "fill") {
			flood_fill(&img,color);
		} else if (cmd == "point") {
			in_file >> point.x;
			in_file >> point.y;
			plot_point(&img,point.x,point.y,color,noisy,perterb);
		} else if (cmd == "circle") {
			in_file >> rng;
			explode_range(&rng,&range);
			circle.a = (int)floor(random_uniform_range(range));

			in_file >> rng;
			explode_range(&rng,&range);
			circle.b = (int)floor(random_uniform_range(range));

			in_file >> rng;
			explode_range(&rng,&range);
			circle.r = (int)floor(random_uniform_range(range));

			bresenham_circle(&img,circle.a,circle.b,circle.r,color,noisy,perterb);
		} else if (cmd == "ellipse") {
			in_file >> rng;
			explode_range(&rng,&range);
			ellipse.a = (int)floor(random_uniform_range(range));

			in_file >> rng;
			explode_range(&rng,&range);
			ellipse.b = (int)floor(random_uniform_range(range));

			in_file >> rng;
			explode_range(&rng,&range);
			ellipse.rx = (int)floor(random_uniform_range(range));

			in_file >> rng;
			explode_range(&rng,&range);
			ellipse.ry = (int)floor(random_uniform_range(range));

			in_file >> rng;
			explode_range(&rng,&range);
			ellipse.angle = (int)floor(random_uniform_range(range));

			//bresenham_ellipse(&img,ellipse.a,ellipse.b,ellipse.rx,ellipse.ry,ellipse.angle,color,noisy,perterb);
			bezier_ellipse(&img,ellipse.a,ellipse.b,ellipse.rx,ellipse.ry,ellipse.angle,color,noisy,perterb);
		} else if (cmd == "bezier") {
			in_file >> bezier.p1.x;
			in_file >> bezier.p1.y;

			in_file >> bezier.p2.x;
			in_file >> bezier.p2.y;

			in_file >> bezier.p3.x;
			in_file >> bezier.p3.y;

			in_file >> bezier.p4.x;
			in_file >> bezier.p4.y;

			debug("(%d,%d) (%d,%d) (%d,%d) (%d,%d)",bezier.p1.x,bezier.p1.y,
			bezier.p2.x,bezier.p2.y,bezier.p3.x,bezier.p3.y,bezier.p4.x,bezier.p4.y);

			bezier_curve(&img,bezier,color,noisy,perterb);
		} else if (cmd == "background") {
			in_file >> bg_noise;
			background_noise(&img,bg_noise,color);
		} else if (cmd == "noisy") {
			in_file >> noisy;
		} else if (cmd == "perterb") {
			in_file >> perterb;
		} else if (cmd == "color") {
			in_file >> r >> g >> b;
			color.r = r;
			color.g = g;
			color.b = b;
		}

		// read the next command from the file
		in_file >> cmd;
	}

	// dump the created image to the output file
	image_write(out_filename,&img);

	// cleanup the program
	terminate();
	return 0;
}

void setup(int* argc, char **argv) {
	// check the lenght of the cmd line
	if (*argc != 2)
		cerr << USAGE_MESSAGE << endl;

	// seed the rng
	srand(clock());

	// open the input file
	in_file.open(argv[1]);
	if (!in_file.is_open()) {
		cerr << "Failed to open file " << argv[1] << " for input!" << endl;
		terminate();
	}
}

void terminate() {
	in_file.close();

	// output the number of edge pixels that were drawn
	cout << "edge pixels drawn = " << edge_pxls_drawn << endl;

	exit(0);
}

// this function processes a range read in from a file
void explode_range(string* s, range_t* r) {
	int i = s->find("-");

	if (i == string::npos) {
		r->low = r->high = atoi(s->c_str());
		return;
	}

	r->low = atoi(s->substr(0,i).c_str());
	s->erase(0,i+1);
	r->high = atoi(s->c_str());
}

// create a uniform random number [0..1)
double random_uniform() {
	return rand() / (float)RAND_MAX;
}

// create a uniform random number in a range [low..high)
double random_uniform_range(range_t &r) {
	return (random_uniform() * (r.high - r.low)) + r.low;
}

// computed using polar method, throw away second result
double gauss_random(double mu, double sigma) {
	double u1, u2, v1, v2, s, z1;

	do {
		u1 = random_uniform();
		u2 = random_uniform();

		v1 = (u1 * 2) - 1;
		v2 = (u2 * 2) - 1;

		s = (v1 * v1) + (v2 * v2);
	} while( s > 1.0 );

	z1 = v1 * pow( ((-2.0 * log(s))/s), 0.5 );

	return (z1 * mu) + sigma;
}

// rotates a point by a number of degrees around another point
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

// blasts an image buffer with a pixel color (naive implementation)
void flood_fill(img_t* img, const color_t &c) {
	for (int y=0; y < img->height; y++) {
		for (int x=0; x < img->width; x++) {
			img->pixels[y][x].rgba = c.rgba;
		}
	}
}

// draws a point with the correct perterbation and noise
void plot_point(img_t* img, uint_t x, uint_t y, const color_t &c, double noisy=0.0, int perterb=0) {
	double draw = 0.0;
	int offset  = 0;

	if (y > img->height-1)
		return;
	if (x > img->width-1)
		return;

	if (perterb > 0) {
		x += (int)floor(gauss_random(double(perterb),0.0));
		y += (int)floor(gauss_random(double(perterb),0.0));
	}

	if (noisy > 0.0) {
		draw = random_uniform();

		debug("%f,%f",draw,noisy);
		if (fabs(draw) < noisy)
			img->pixels[y][x].rgba = c.rgba;
	} else
		img->pixels[y][x].rgba = c.rgba;

	edge_pxls_drawn++;
}

// plot out a bezier curve
void bezier_curve(img_t* img, bezier_t &b, const color_t &c, double noisy=0.0, int perterb=0) {
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

		//debug("%0.3f: (%d,%d) (%d,%d)",t,p1.x,p1.y,p2.x,p2.y);
		bresenham_line(img,p1,p2,c,noisy,perterb);

		p1.x = p2.x;
		p1.y = p2.y;
	}
	bresenham_line(img,p1,b.p4,c,noisy,perterb);
}

// draw an ellipse with bezier curves
void bezier_ellipse(img_t* img, uint_t a, uint_t b, uint_t rx, uint_t ry, uint_t angle, const color_t &c, double noisy=0.0, int perterb=0) {
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

	bezier_curve(img,b1,c,noisy,perterb);
	bezier_curve(img,b2,c,noisy,perterb);
	bezier_curve(img,b3,c,noisy,perterb);
	bezier_curve(img,b4,c,noisy,perterb);
}

// fast version of bresenham's line drawing algorithm
void bresenham_line(img_t* img, point_2d_t p1, point_2d_t p2, const color_t &c, double noisy=0.0, int perterb=0) {
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

		plot_point(img, p.x, p.y, c, noisy, perterb);

		error -= deltay;
		if (error < 0) {
			error += deltax;
			y += stepy;
		}
	}
}

// a fast version of the bresenham circle drawing algorithm
void bresenham_circle(img_t* img, uint_t a, uint_t b, uint_t r, const color_t &c, double noisy=0.0, int perterb=0) {
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

		plot_point(img, a + x, b + y, c, noisy, perterb);
		plot_point(img, a + x, b - y, c, noisy, perterb);
		plot_point(img, a - x, b + y, c, noisy, perterb);
		plot_point(img, a - x, b - y, c, noisy, perterb);
		plot_point(img, a + y, b + x, c, noisy, perterb);
		plot_point(img, a + y, b - x, c, noisy, perterb);
		plot_point(img, a - y, b + x, c, noisy, perterb);
		plot_point(img, a - y, b - x, c, noisy, perterb);
	}
}

// a function to draw an ellipse in a bresenham style
// based on this paper: http://homepage.smc.edu/kennedy_john/belipse.pdf
void bresenham_ellipse(img_t* img, uint_t a, uint_t b, uint_t rx, uint_t ry, uint_t angle, const color_t &c, double noisy=0.0, int perterb=0) {
	int x, y, dx, dy, error, stop_x, stop_y, rx_2_sq, ry_2_sq, tx, ty, tnx, tny;
	double theta,cos_theta,sin_theta;

	theta     = (angle % 360) * ((2 * M_PI) / 360.0);
	cos_theta = cos(theta);
	sin_theta = sin(theta);
	rx_2_sq   = 2 * rx * rx;
	ry_2_sq   = 2 * ry * ry;
	x         = rx;
	y         = 0;
	dx        = ry * ry * (1 - 2 * rx);
	dy        = rx * rx;
	error     = 0;
	stop_x    = ry_2_sq * rx;
	stop_y    = 0;

	// first set of points
	while (stop_x >= stop_y) {
		tx  = x * cos_theta - y * sin_theta;
		ty  = x * sin_theta + y * cos_theta;
		plot_point(img, a - ty, b - tx, c, noisy, perterb);
		plot_point(img, a + tx, b - ty, c, noisy, perterb);
		plot_point(img, a - ty, b + tx, c, noisy, perterb);
		plot_point(img, a + tx, b + ty, c, noisy, perterb);

		y++;
		stop_y += rx_2_sq;
		error  += dy;
		dy     += rx_2_sq;

		if ((2 * error + dx) > 0) {
			x--;
			stop_x -= ry_2_sq;
			error  += dx;
			dx     += ry_2_sq;
		}
	}

	// second set of points
	x      = 0;
	y      = ry;
	dx     = ry * ry;
	dy     = rx * rx * (1 - 2 * ry);
	error  = 0;
	stop_x = 0;
	stop_y = rx_2_sq * ry;

	while (stop_x <= stop_y) {
		tx = x * cos_theta - y * sin_theta;
		ty = x * sin_theta + y * cos_theta;
		plot_point(img, a - ty, b - tx, c, noisy, perterb);
		plot_point(img, a + tx, b - ty, c, noisy, perterb);
		plot_point(img, a - ty, b + tx, c, noisy, perterb);
		plot_point(img, a + tx, b + ty, c, noisy, perterb);

		x++;
		stop_x += ry_2_sq;
		error  += dx;
		dx     += ry_2_sq;

		if ((2 * error + dy) > 0) {
			y--;
			stop_y -= rx_2_sq;
			error  += dy;
			dy     += rx_2_sq;
		}
	}
}

// draw randomly distributed gaussian background noisef
void background_noise(img_t* img, double pct_coverage, const color_t & c) {
	int edge_pxls,x,y;
	double thrsh;

	edge_pxls = 0;
	for (y=0; y < img->height; y++) {
		for(x=0; x < img->width; x++) {
			if (img->pixels[y][x].rgba != 0)
				edge_pxls++;
		}
	}
	edge_pxls *= pct_coverage;

	debug("%f %d",pct_coverage,edge_pxls);

	for (int i=0; i < edge_pxls; i++) {
		do {
			x = (int)floor((double)img->width * random_uniform());
			y = (int)floor((double)img->height * random_uniform());
		} while (img->pixels[y][x].rgba != 0);

		plot_point(img, x, y, c);
	}
}

