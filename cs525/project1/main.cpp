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

#define WINDOW_X 401
#define WINDOW_Y 401
#define WINDOW_NAME "Homework 1 -- Sine Wave"
#define PI 3.14159265

struct point {
	float x;
	float y;
};

struct line_stats {
	float min;
	float max;
	float avg;
	float count;
};

void display_func(void);
void keyboard_func(unsigned char, int, int);
void terminate(void);
float line_length(point , point );

line_stats sin_lines;
line_stats cos_lines;

int main(int argc, char* argv[]) {

	glutInit(&argc,argv);

	init_setup(WINDOW_X,WINDOW_Y,WINDOW_NAME);

	// setup the stats structures
	sin_lines.min = 10000.0; // large value
	sin_lines.max = sin_lines.avg = sin_lines.count = 0.0;
	cos_lines.min = 10000; // large value
	cos_lines.max = cos_lines.avg = cos_lines.count = 0;

	glutDisplayFunc(display_func);
	glutKeyboardFunc(keyboard_func);
	glutMainLoop();

	return 0;
}

void display_func(void) {
	int i;
	float ll;
	point p1,p2;

	// white background
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	// draw the axis
	glColor3f(0.0,0.0,0.0);
	glBegin(GL_LINES);
		glVertex2i(0,200);
		glVertex2i(401,200);
		glVertex2i(200,0);
		glVertex2i(200,401);
	glEnd();

	// draw the sin
	p1.x = p2.x = 0.0;
	p1.y = p2.y = 200.5 + 200.5 *
			sin(((PI * (p2.x - 401)) / 200.5) + PI);
	for(i=0; i < WINDOW_X+1; i++) {
		p2.x++;
		p2.y = 200.5 + 200.5 *
			sin(((PI * (p2.x - 401)) / 200.5) + PI);
		glColor3f(1.0,0.0,0.0);
		glBegin(GL_LINES);
			glVertex2f(p1.x,p1.y);
			glVertex2f(p2.x,p2.y);
		glEnd();

		ll = line_length(p1,p2);

		if (sin_lines.min > ll)
			sin_lines.min = ll;
		if (sin_lines.max < ll)
			sin_lines.max = ll;
		sin_lines.avg = (sin_lines.avg + ll) / 2;
		sin_lines.count++;

		p1.x = p2.x;
		p1.y = p2.y;
	}

	// draw the cos
	p1.x = p2.x = 0.0;
	p1.y = p2.y = 200.5 + 200.5 *
			cos(((PI * (p2.x - 401)) / 200.5) + ((3 * PI) / 2));
	for(i=0; i < WINDOW_X+1; i++) {
		p2.x++;
		p2.y = 200.5 + 200.5 *
			cos(((PI * (p2.x - 401)) / 200.5) + ((3 * PI) / 2));
		glColor3f(0.0,0.0,1.0);
		glBegin(GL_LINES);
			glVertex2f(p1.x,p1.y);
			glVertex2f(p2.x,p2.y);
		glEnd();

		ll = line_length(p1,p2);

		if (cos_lines.min > ll)
			cos_lines.min = ll;
		if (cos_lines.max < ll)
			cos_lines.max = ll;
		cos_lines.avg = (cos_lines.avg + ll) / 2;
		cos_lines.count++;
		p1.x = p2.x;
		p1.y = p2.y;
	}

	// flush the buffer
	glFlush();
}

void keyboard_func (unsigned char c, int x, int y) {
	switch (c) {
		case 'Q':
		case 'q': {
			terminate();
			exit(0);
		} break;
	}

}

void terminate(void) {
	// dump the sin & cos line stats
	cout << "sin stats:" << endl << "\tcount: " << sin_lines.count << endl
		 << "\tmin:" << sin_lines.min << endl << "\tmax:" << sin_lines.max
		 << endl << "\tavg:" << sin_lines.avg << endl << endl;
	cout << "cos stats:" << endl << "\tcount: " << cos_lines.count << endl
		 << "\tmin:" << cos_lines.min << endl << "\tmax:" << cos_lines.max
		 << endl << "\tavg:" << cos_lines.avg << endl << endl;
}

float line_length(point p1, point p2) {
	return sqrt(((p1.x - p2.x)*(p1.x-p2.x)) +
		((p1.y - p2.y)*(p1.y - p2.y)));
}
