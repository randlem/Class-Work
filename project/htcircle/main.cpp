#define DEBUG		1
#define WINDOW_X	501
#define WINDOW_Y	501
#define WINDOW_NAME "Circle Hough Transform"
#define BIN_THRLD   0.6
#define CIRCLE_PTS  64

#define IMG(x,y)     img.pixels[(y)][(x)]
#define IMG_MAP(x,y) img_map[(y) * img.width + (x)]
#define ACCUM(a,b,r) accum[(a)][(b)][(r)]
#define RANGE(rng)   (double)(rng.high - rng.low)
//((((a) * a_bins + (b)) * b_bins) + (r))]

#include <iostream>
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <math.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include "cs_456_250_setup.h"
#include "util.h"
#include "gfxutil.h"
#include "imgutil.h"

typedef struct {
	double a;
	double b;
	double r;
	uint_t bin_cnt;
} circle_t;

typedef struct {
	double low;
	double high;
} range_t;

typedef struct {
	double avg_bin_cnt;
	uint_t max_bin_cnt;
	uint_t filled_bins;
} stats_t;

//const string in_file = "edge.png";
//const string in_file = "single_circle.png";
//const string in_file = "many_circle.png";
const string in_file  = "with_noise.png";
const string out_file = "out.png";

const range_t r_range = {5.0, 125.0};
const range_t a_range = {-r_range.high, 250.0 + r_range.high};
const range_t b_range = {-r_range.high, 250.0 + r_range.high};

img_t     img;
uchar_t*  img_map   = NULL;
uint_t*** accum	    = NULL;
stats_t   stats;
circle_t* circles   = NULL;
uint_t    num_circles;

uint_t a_bins = 0;
uint_t b_bins = 0;
uint_t r_bins = 0;
uint_t accum_size = 0;

void setup(int*, char**);
void terminate();
void display_func(void);
void keyboard_func(uchar_t, int, int);
void collect_stats();

void draw_circle(const circle_t&, const uint_t, const uint_t, const color_gl_t&);

void translate_img_to_binary(img_t*);
void allocate_accum();
void ht_circle();
int identify_circle();

int main (int argc, char *argv[]) {
	setup(&argc,argv);

	if (!image_read(in_file,&img))
		return 1;

	translate_img_to_binary(&img);
	ht_circle();
	collect_stats();
	num_circles = identify_circle();
	debug("Num. circles = %d",num_circles);

	// run the OGL main loop
	glutMainLoop();

	return 0;
}

void setup(int* argc, char **argv) {
	img_map = NULL;
	accum = NULL;

	memset(&stats,0,sizeof(stats_t));

	glutInit(argc,argv);
	init_setup(WINDOW_X,WINDOW_Y,WINDOW_NAME);
	glutDisplayFunc(display_func);
	glutKeyboardFunc(keyboard_func);
}

void terminate() {
	if (img_map != NULL) {
		delete [] img_map;
		img_map = NULL;
	}

	if (accum != NULL) {
		delete [] accum;
		accum = NULL;
	}
}

void display_func() {
	int y,x,i;

	// blank the background
	glClearColor(BLACK.r, BLACK.g, BLACK.b, BLACK.a);
	glClear(GL_COLOR_BUFFER_BIT);

	// draw the frame traces
	glColor4f(GLC_ORANGE.r, GLC_ORANGE.g, GLC_ORANGE.b, GLC_ORANGE.a);
	glBegin(GL_LINES);
		glVertex2i(251, 0);
		glVertex2i(251, WINDOW_Y);
		glVertex2i(0, 251);
		glVertex2i(WINDOW_X, 251);
	glEnd();

	// draw the original image in the top left corner
	glBegin(GL_POINTS);
	for (y=0; y < img.height; y++) {
		for (x=0; x < img.width; x++) {
			if (img_map[y * img.width + x] != 0) {
				glColor4f(GLC_WHITE.r, GLC_WHITE.g, GLC_WHITE.b, GLC_WHITE.a);
				glVertex2i(x, WINDOW_Y - y);
			}
		}
	}
	glEnd();
	print_string(254, 3, GLUT_BITMAP_9_BY_15, GLC_LT_BLUE, "File: %s", in_file.c_str());

	// draw the accum array

	// draw the original image with detected circles overlayed
	glBegin(GL_POINTS);
	for (y=0; y < img.height; y++) {
		for (x=0; x < img.width; x++) {
			if (img_map[y * img.width + x] != 0) {
				glColor4f(GLC_WHITE.r, GLC_WHITE.g, GLC_WHITE.b, GLC_WHITE.a);
				glVertex2i(x + 250, 250 - y);
			}
		}
	}
	glEnd();

	for (i=0; i < num_circles; i++) {
		draw_circle(circles[i],250,250,GLC_ORANGE);
	}

	// draw some information in the top right corner
	print_string(WINDOW_Y - 18, 254, GLUT_BITMAP_9_BY_15, GLC_WHITE,
		"Filled Bins = %d", stats.filled_bins);
	print_string(WINDOW_Y - 36, 254, GLUT_BITMAP_9_BY_15, GLC_WHITE,
		"Max Bin Cnt = %d", stats.max_bin_cnt);
	print_string(WINDOW_Y - 54, 254, GLUT_BITMAP_9_BY_15, GLC_WHITE,
		"Avg Bin Cnt = %0.2f", stats.avg_bin_cnt);

	glFlush();
	glutSwapBuffers();
}

void keyboard_func(uchar_t c, int x, int y) {
	switch (c) {
		case 'q':
		case 'Q': {
			terminate();
			exit(0);
		} break;
	}

	glutPostRedisplay();
}

void collect_stats() {
	int a_bin,b_bin,r_bin,curr;

	for (a_bin=0; a_bin < a_bins; a_bin++) {
		for (b_bin=0; b_bin < b_bins; b_bin++) {
			for (r_bin=0; r_bin < r_bins; r_bin++) {
				curr = ACCUM(a_bin,b_bin,r_bin);
				if (curr > 0) {
					stats.filled_bins++;
					if (stats.max_bin_cnt < curr)
						stats.max_bin_cnt = curr;
					stats.avg_bin_cnt += curr;
				}
			}
		}
	}
	stats.avg_bin_cnt = stats.avg_bin_cnt / stats.filled_bins;
}

void draw_circle(const circle_t& c, const uint_t offset_x, const uint_t offset_y, const color_gl_t & cl) {
	double theta;

	debug("Drawing circle (%0.2f,%0.2f,%0.2f) %d", c.a, c.b, c.r, c.bin_cnt);

	glBegin(GL_LINE_LOOP);
		glColor4f(cl.r,cl.g,cl.b,cl.a);
		for(theta=0; theta < 2 * M_PI; theta += (M_PI / CIRCLE_PTS)) {
			glVertex2i(
				(int)(c.a + cos(theta) * c.r) + offset_x,
				offset_y - (int)(c.b + sin(theta) * c.r)
			);
		}
	glEnd();
}

void translate_img_to_binary(img_t* img) {
	int y,x,map_size;

	map_size = img->height * img->width;
	img_map = new uchar_t[map_size];
	memset(img_map,0,map_size * sizeof(uchar_t));

	for (y=0; y < img->height; y++) {
		for (x=0; x < img->width; x++) {
			img_map[y * img->width + x] = img->pixels[y][x].rgba;
		}
	}
}

void allocate_accum() {
	int i,j;

	a_bins     = (uint_t)RANGE(a_range);
	b_bins     = (uint_t)RANGE(b_range);
	r_bins     = (uint_t)RANGE(r_range);
	accum_size = a_bins * b_bins * r_bins;

	accum      = new uint_t**[a_bins];
	for(i=0; i < a_bins; i++) {
		accum[i] = new uint_t*[b_bins];
		for(j=0; j < b_bins; j++) {
			accum[i][j] = new uint_t[r_bins];
			memset(accum[i][j],0,r_bins*sizeof(uint_t));
		}
	}

	debug("Accum: size = %d",accum_size);
	debug("A: range = (%0.2f,%0.2f) bins = %d",a_range.low,a_range.high,a_bins);
	debug("B: range = (%0.2f,%0.2f) bins = %d",b_range.low,b_range.high,b_bins);
	debug("R: range = (%0.2f,%0.2f) bins = %d",r_range.low,r_range.high,r_bins);
}

void ht_circle() {
	int y,x,a_bin,b_bin,r_bin;
	double a,b,r,theta,a_pitch,b_pitch,r_pitch;
	int min_a_bin = 100000, min_b_bin = 100000, min_r_bin = 100000;
	double min_a = 100000, min_b = 100000;

	allocate_accum();
	a_pitch = RANGE(a_range) / a_bins;
	b_pitch = RANGE(b_range) / b_bins;
	r_pitch = RANGE(r_range) / r_bins;

	debug("%0.2f %0.2f %0.2f",a_pitch,b_pitch,r_pitch);

	for (y=0; y < img.height; y++) {
		for (x=0; x < img.width; x++) {
			if (IMG_MAP(x,y) > 0) {
				for (r_bin=(int)r_range.low; r_bin < r_bins; r_bin++) {
					for (theta=0.0; theta < 360.0; theta++) {
						r = r_bin * r_pitch;
						a = x + (r * cos((theta * 2.0 * M_PI) / 360.0));
						b = y + (r * sin((theta * 2.0 * M_PI) / 360.0));
						a_bin = (int)(a / a_pitch) + (int)fabs(a_range.low);
						b_bin = (int)(b / b_pitch) + (int)fabs(b_range.low);
						ACCUM(a_bin, b_bin, r_bin)++;

						if (a_bin < min_a_bin)
							min_a_bin = a_bin;
						if (b_bin < min_b_bin)
							min_b_bin = b_bin;
						if (r_bin < min_r_bin)
							min_r_bin = r_bin;
						if (a < min_a)
							min_a = a;
						if (b < min_b)
							min_b = b;
					}
				}
			}
		}
	}

	debug("%d %d %d %0.2f %0.2f",min_a_bin, min_b_bin, min_r_bin, min_a, min_b);
}

int identify_circle() {
	int threshold,a_bin,b_bin,r_bin,circles_cnt,max_circles;
	double a_pitch, b_pitch, r_pitch;

	a_pitch     = RANGE(a_range) / a_bins;
	b_pitch     = RANGE(b_range) / b_bins;
	r_pitch     = RANGE(r_range) / r_bins;
	threshold   = (int)ceil(stats.max_bin_cnt * BIN_THRLD);
	circles_cnt = 0;
	max_circles = (int)(stats.filled_bins * 0.01);
	circles     = new circle_t[max_circles];
	if (circles == NULL) {
		debug("Failed to allocate circles array.");
		return 0;
	}
	memset(circles,0,max_circles * sizeof(circle_t));

	for (a_bin=0; a_bin < a_bins; a_bin++) {
		for (b_bin=0; b_bin < b_bins; b_bin++) {
			for (r_bin=0; r_bin < r_bins; r_bin++) {
				if (ACCUM(a_bin,b_bin,r_bin) >= threshold) {
					circles[circles_cnt].a       = (a_bin - fabs(a_range.low)) * a_pitch;
					circles[circles_cnt].b       = (b_bin - fabs(b_range.low)) * b_pitch;
					circles[circles_cnt].r       = r_bin * r_pitch;
					circles[circles_cnt].bin_cnt = ACCUM(a_bin,b_bin,r_bin);
					circles_cnt++;
					max_circles--;
				}
				if (max_circles < 0) {
					debug("Hit max circles threshold.");
					goto end;
				}
			}
		}
	}

	end:
	return circles_cnt;
}
