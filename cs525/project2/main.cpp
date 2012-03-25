/*******************************************************************************
* Project 2 -- Bresenham's Snowflake
* Mark Randles CS525
* 2009-02-13
*
* SHORT DESC: This program draws a snowflake drawing lines pixel-by-pixel.  Each
*  pixel is actually 5x5 square of screen pixels, centered on the actual pixel.
*  There are a number of different functionalities available to the user.
*
* INTERFACE CONTROLS:
*  Left Mouse Click: Draw the whole snowflake in the current color and alpha
*		transparency.
*  Right Mouse Click: Draw the whole snowflake line-by-line in the current color
*		and alpha transparency.
*  a: Decrease alpha of current color by 10%
*  A: Increase alpha of current color by 10%
*  W,w: Change foreground (draw) color to white.  Does not reset alpha.
*  R,r: Change foreground (draw) color to red.  Does not reset alpha.
*  G,g: Change foreground (draw) color to green.  Does not reset alpha.
*  B,b: Change foreground (draw) color to blue.  Does not reset alpha.
*  Q,q: Quit the program.
*
*******************************************************************************/

#include <iostream>
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <GL/glut.h>

#include "cs_425_525_setup.h"

#define WINDOW_X 600
#define WINDOW_Y 600
#define WINDOW_NAME "Homework 2 -- Bresenham Snowflake"
#define PI 3.14159265

typedef struct {
	int x;
	int y;
} point;

typedef struct {
	point p1;
	point p2;
} line;

typedef struct {
	float r;
	float g;
	float b;
	float a;
} color;

typedef struct {
	int state; // 0 is all lines, 1 is line-by-line (mouse input)
	int curr_line;
} draw_state;

static color BACKGROUND = { 0.0, 0.0, 0.0, 1.0 };
static color FOREGROUND = { 1.0, 1.0, 1.0, 1.0 };

// the snowflake
const int NUM_SNOWFLAKE_LINES = 20;
const line SNOWFLAKE[20] = {
	{{50,300},{550,300}},
	{{50,365},{175,300}},
	{{50,240},{175,300}},
	{{550,365},{425,300}},
	{{550,240},{425,300}},
	{{300,50},{300,550}},
	{{215,550},{300,425}},
	{{390,550},{300,425}},
	{{215,50},{300,175}},
	{{390,50},{300,175}},
	{{50,550},{550,50}},
	{{50,490},{175,425}},
	{{110,550},{175,425}},
	{{490,50},{425,175}},
	{{550,110},{425,175}},
	{{50,50},{550,550}},
	{{50,110},{175,175}},
	{{110,50},{175,175}},
	{{490,550},{425,425}},
	{{550,490},{425,425}}
};

draw_state curr_draw_state = {0,0};

void display_func(void);
void keyboard_func(unsigned char, int, int);
void mouse_func(int, int, int, int);
void terminate(void);

void prep_window(void);
void draw_snowflake(void);
void draw_snowflake_next(void);

void bresenham(point, point );
void draw_big_pix(point);

int main(int argc, char* argv[]) {

	glutInit(&argc,argv);

	init_setup(WINDOW_X,WINDOW_Y,WINDOW_NAME);
	glutDisplayFunc(display_func);
	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(mouse_func);
	glutMainLoop();

	return 0;
}

void display_func(void) {
	// switch on the current draw state and run the right draw procedure
	switch (curr_draw_state.state) {
		case 1: { // draw the next snowflake line
			draw_snowflake_next();
		} break;
		case 0:
		default: { // draw the whole snowflake
			prep_window();
			draw_snowflake();
		} break;
	}

	// flush the buffer
	glFlush();
}

void keyboard_func (unsigned char c, int x, int y) {
	// switch on the keyboard character pressed
	switch (c) {
		case 'W': // change draw color to white
		case 'w': {
			FOREGROUND.r = 1.0;
			FOREGROUND.g = 1.0;
			FOREGROUND.b = 1.0;
		} break;
		case 'R': // change draw color to red
		case 'r': {
			FOREGROUND.r = 1.0;
			FOREGROUND.g = 0.0;
			FOREGROUND.b = 0.0;
		} break;
		case 'G': // change draw color to green
		case 'g': {
			FOREGROUND.r = 0.0;
			FOREGROUND.g = 1.0;
			FOREGROUND.b = 0.0;
		} break;
		case 'B': // change draw color to blue
		case 'b': {
			FOREGROUND.r = 0.0;
			FOREGROUND.g = 0.0;
			FOREGROUND.b = 1.0;
		} break;
		case 'A': { // up the alpha and keep in bounds
			FOREGROUND.a += 0.1;
			if (FOREGROUND.a > 1.0)
				FOREGROUND.a = 1.0;
		} break;
		case 'a': { // down the alpha and keep in bounds
			FOREGROUND.a -= 0.1;
			if (FOREGROUND.a < 0)
				FOREGROUND.a = 0.0;
		} break;
		case 'Q': // quit the program
		case 'q': {
			terminate();
			exit(0);
		} break;
	}

	// force a redraw
	glutPostRedisplay();
}

void mouse_func (int button, int state, int x, int y) {
	// capture the left button press and set the proper state, clear the line counter
	if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) {
		curr_draw_state.state = 0;
		curr_draw_state.curr_line = 0;
	}

	// capture the right button press and set the proper state
	if ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN)) {
		if (curr_draw_state.state != 1)
			curr_draw_state.state = 1;
	}

	// force a redraw
	glutPostRedisplay();
}

void terminate(void) {

}

void prep_window(void) {
	// set the background color and clear the screen
	glClearColor(BACKGROUND.r, BACKGROUND.g, BACKGROUND.b, BACKGROUND.a);
	glClear(GL_COLOR_BUFFER_BIT);
}

void draw_snowflake(void) {
	// loop through all snowflake lines and draw them.
	for(int i=0; i < NUM_SNOWFLAKE_LINES; i++) {
		bresenham(SNOWFLAKE[i].p1,SNOWFLAKE[i].p2);
	}
}

void draw_snowflake_next(void) {
	// clear the screen if we need to draw the first line
	if (curr_draw_state.curr_line == 0) {
		prep_window();
	}

	// draw the next line.
	bresenham(SNOWFLAKE[curr_draw_state.curr_line].p1,
		SNOWFLAKE[curr_draw_state.curr_line].p2);

	// increment the current line and wrap if over the number of lines in snowflake
	curr_draw_state.curr_line++;
	curr_draw_state.curr_line %= NUM_SNOWFLAKE_LINES;
}

void bresenham(point p1, point p2) {
	int deltax = (p2.x - p1.x), deltay = (p2.y - p1.y),
		dfa, dfb, error,
		steep, stepx, stepy,
		x, y, t;
	point p;

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
	stepx = 5;
	stepy = (p1.y < p2.y) ? 5 : -5;

	// draw the line
	y = p1.y;
	for (x=p1.x; x != p2.x; x+=stepx) {
		if (steep) {
			p.x = y;
			p.y = x;
		} else {
			p.x = x;
			p.y = y;
		}

		draw_big_pix(p);

		error -= deltay;
		if (error < 0) {
			error += deltax;
			y += stepy;
		}
	}
}

void draw_big_pix(point p) {
	glColor4f(FOREGROUND.r, FOREGROUND.g, FOREGROUND.b,FOREGROUND.a);
	glBegin(GL_POINTS);
		for (int i=p.x-2; i <= p.x+2; i++) {
			for (int j=p.y-2; j <= p.y+2; j++) {
				glVertex2i(i,j);
			}
		}
	glEnd();
}
