#define DEBUG		1
#define WINDOW_X	501
#define WINDOW_Y	501
#define WINDOW_NAME "Line Hough Transform"

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
	float r;
	float theta;
} line_t;

//const string in_file = "edge.png";
const string in_file = "sqr1can1.png";
const string out_file = "out.png";

img_t    img;
uchar_t* img_map   = NULL;
uint_t*  bin_cnt   = NULL;
int      bin_cnt_x;
int      bin_cnt_y;
float    bin_pitch_x;
float    bin_pitch_y;
float    bin_max_x;
float    bin_max_y;

float    computed_max_y;

void setup(int*, char**);
void terminate();
void display_func(void);
void keyboard_func(uchar_t, int, int);

void translate_img_to_binary(img_t*);
void hough_transform();
void identify_lines(int, line_t*);

int main (int argc, char *argv[]) {
	setup(&argc,argv);

	if (!loadImage(in_file,&img))
		return 1;

	translate_img_to_binary(&img);
	hough_transform();

	// run the OGL main loop
	glutMainLoop();

	return 0;
}

void setup(int* argc, char **argv) {
	img_map = NULL;

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

	if (bin_cnt != NULL) {
		delete [] bin_cnt;
		bin_cnt = NULL;
	}
}

void display_func() {
	int i,y,x,max_cnt,filled_bins,line_cnt,x0,x1,y0,y1;
	float avg_cnt;
	line_t *lines;

	// blank the background
	glClearColor(BLACK.r, BLACK.g, BLACK.b, BLACK.a);
	glClear(GL_COLOR_BUFFER_BIT);

	// draw the frame traces
	glColor4f(ORANGE.r, ORANGE.g, ORANGE.b, ORANGE.a);
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
				glColor4f(WHITE.r, WHITE.g, WHITE.b, WHITE.a);
				glVertex2i(x, WINDOW_Y - y);
			}
		}
	}
	glEnd();
	print_string(254, 3, GLUT_BITMAP_9_BY_15, LT_BLUE, "File: %s", in_file.c_str());

	// draw the hough space in the bottom left corner
	max_cnt = 0; avg_cnt = 0.0; filled_bins = 0;
	for (y=0; y < bin_cnt_y; y++) {
		for (x=0; x < bin_cnt_x; x++) {
			if (bin_cnt[y * bin_cnt_x + x] > 0) {
				avg_cnt += bin_cnt[y * bin_cnt_x + x];
				filled_bins++;
				if (bin_cnt[y * bin_cnt_x + x] > max_cnt)
					max_cnt = bin_cnt[y * bin_cnt_x + x];
			}
		}
	}
	avg_cnt /= (float)filled_bins;

	glBegin(GL_POINTS);
	for (y=0; y < bin_cnt_y; y++) {
		for (x=0; x < bin_cnt_x; x++) {
			glColor4f(
				(float)bin_cnt[y * bin_cnt_x + x]/max_cnt,
				(float)bin_cnt[y * bin_cnt_x + x]/max_cnt,
				(float)bin_cnt[y * bin_cnt_x + x]/max_cnt,
				1.0
			);
			glVertex2i(x, y);
		}
	}
	glEnd();

	// redraw the original image and overlay it with lines identified in yellow
	glBegin(GL_POINTS);
	for (y=0; y < img.height; y++) {
		for (x=0; x < img.width; x++) {
			glColor4f(
				img.pixels[y][x].r,
				img.pixels[y][x].g,
				img.pixels[y][x].b,
				1.0
			);
			glVertex2i(251 + x, 249 - y);
		}
	}
	glEnd();

	line_cnt = 50;
	lines = new line_t[line_cnt];
	memset(lines,0,line_cnt*sizeof(line_t));
	identify_lines(line_cnt,lines);
	for (i=0; i < line_cnt; i++) {
		if (abs(lines[i].theta) >= (M_PI / 4)) {
			x0 = 0; x1 = 249;
			y0 = (float)(x0 * (-cos(lines[i].theta) / sin(lines[i].theta))) + (lines[i].r / sin(lines[i].theta));
			y1 = (float)(x1 * (-cos(lines[i].theta) / sin(lines[i].theta))) + (lines[i].r / sin(lines[i].theta));
		} else {
			y0 = 0; y1 = 249;
			x0 = (float)(lines[i].r - y0 * sin(lines[i].theta)) / cos(lines[i].theta);
			x1 = (float)(lines[i].r - y1 * sin(lines[i].theta)) / cos(lines[i].theta);
		}

		debug ("Drawing line (%0.4f,%0.4f) as (%d,%d),(%d,%d)",
			lines[i].r, lines[i].theta, x0, y0, x1, y1);

		glBegin(GL_LINES);
			glColor4f(YELLOW.r,YELLOW.g,YELLOW.b,YELLOW.a);
			glVertex2i(x0 + 250,y0);
			glVertex2i(x1 + 250,y1);
		glEnd();
	}

	// draw some information in the top right corner
	print_string(WINDOW_Y - 18, 254, GLUT_BITMAP_9_BY_15, WHITE,
		"Bin Pitch X = %0.2f", bin_pitch_x);
	print_string(WINDOW_Y - 36, 254, GLUT_BITMAP_9_BY_15, WHITE,
		"Bin Pitch Y = %0.2f", bin_pitch_y);
	print_string(WINDOW_Y - 54, 254, GLUT_BITMAP_9_BY_15, WHITE,
		"Avg Bin Cnt = %0.2f", avg_cnt);
	print_string(WINDOW_Y - 72, 254, GLUT_BITMAP_9_BY_15, WHITE,
		"Computed Max Y = %0.2f", computed_max_y);
	print_string(WINDOW_Y - 90, 254, GLUT_BITMAP_9_BY_15, WHITE,
		"Avg Bin Cnt = %0.2f", avg_cnt);
	print_string(WINDOW_Y - 108, 254, GLUT_BITMAP_9_BY_15, WHITE,
		"Filled Bins = %d", filled_bins);
	print_string(WINDOW_Y - 126, 254, GLUT_BITMAP_9_BY_15, WHITE,
		"Max Bin Cnt = %d", max_cnt);

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

void hough_transform() {
	int ary_size,y,x,bin_y,bin_x;
	float theta, r;

	bin_max_x = M_PI / 2;
	bin_max_y = sqrt((img.width * img.width) + (img.height * img.height));
	bin_cnt_x = img.width;
	bin_cnt_y = img.height;
	bin_pitch_x = (float)(bin_max_x * 2) / bin_cnt_x;
	bin_pitch_y = (float)(bin_max_y * 2) / bin_cnt_y;
	ary_size = bin_cnt_x * bin_cnt_y;
	computed_max_y = 0.0;

	debug("bin_max_x = %0.2f",bin_max_x);
	debug("bin_max_y = %0.2f",bin_max_y);
	debug("bin_cnt_x = %d",bin_cnt_x);
	debug("bin_cnt_y = %d",bin_cnt_y);
	debug("bin_pitch_x = %0.4f",bin_pitch_x);
	debug("bin_pitch_y = %0.4f",bin_pitch_y);

	bin_cnt = new uint_t[ary_size];
	memset(bin_cnt, 0, sizeof(uint_t) * ary_size);

	for(y=0; y < img.height; y++)
		bin_cnt[y * bin_cnt_x + 125] = 1;

	for(x=0; x < img.width; x++)
		bin_cnt[125 * bin_cnt_x + x] = 1;

	for (y=0; y < img.height; y++) {
		for (x=0; x < img.width; x++) {
			if (img_map[y * img.width + x] > 0) {
				for (theta=bin_max_x; theta >= -bin_max_x; theta-=bin_pitch_x) {
					r = x * cos(theta) + y * sin(theta);
					bin_x = (int)floor(theta / bin_pitch_x) + ((float)bin_cnt_x / 2);
					bin_y = (int)floor((r / bin_pitch_y) + ((float)bin_cnt_y / 2));
					bin_cnt[bin_y * bin_cnt_x + bin_x]++;
					if (computed_max_y < r)
						computed_max_y = r;
				}
			}
		}
	}
}

void identify_lines(int cnt, line_t* lines) {
	int cnt_max,max_i,y,x,i;
	uint_t* bins;
	float r, theta;

	debug("Trying to identify %d lines",cnt);

	bins = new uint_t[bin_cnt_x * bin_cnt_y];
	memcpy(bins,bin_cnt,bin_cnt_x * bin_cnt_y * sizeof(uint_t));

	cnt_max = 0; i = 0;
	while (i < cnt) {
		for (y=0; y < bin_cnt_y; y++) {
			for (x=0; x < bin_cnt_x; x++) {
				if (bins[y * bin_cnt_x + x] > cnt_max) {
					cnt_max = bins[y * bin_cnt_x + x];
					max_i = y * bin_cnt_x + x;
					r = (y - (bin_cnt_y / 2)) * bin_pitch_y;
					theta = (x - (bin_cnt_x / 2)) * bin_pitch_x;
				}
			}
		}
		debug("Identified line %0.2f,%0.2f", r, theta);
		cnt_max = 0;
		lines[i].r = r;
		lines[i].theta = theta;
		bins[max_i] = 0;
		i++;
	}
}
