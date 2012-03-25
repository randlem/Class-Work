#include <iostream>
using std::cout;
using std::endl;
using std::ios;

#include <string>
using std::string;

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <GL/gl.h>
#include <GL/glut.h>

#include "lib.h"
#include "bpm.h"

#define WINDOW_X 800
#define WINDOW_Y 600
#define WINDOW_NAME "Virtual Orchestra 1000 Extreme Pro Edition"
#define PI 3.14159265

#define RANDRANGE(a,b) (rand() % ((b) - (a) + 1) + (a))
#define SWAP(a,b) ((a) ^= (b) ^= (a) ^= (b))

void reshape_handler(int, int);
void init_setup(int, int, char*);
void display_func(void);
void keyboard_func(uchar_t, int, int);
void keyboard_special_func(int, int, int);
void mouse_func(int, int, int, int);
void idle_func(void);

int main(int argc, char* argv[]) {
	// setup data structs
	bpm_init();

	// setup the program
	glutInit(&argc,argv);
	init_setup(WINDOW_X,WINDOW_Y,WINDOW_NAME);

	// setup glut callback functions
	glutDisplayFunc(display_func);
	glutKeyboardFunc(keyboard_func);
	glutSpecialFunc(keyboard_special_func);
	glutMouseFunc(mouse_func);
	glutIdleFunc(idle_func);

	// run the OGL main loop
	glutMainLoop();

	return 0;
}

void reshape_handler(int width, int height) {
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);
}

void init_setup(int width, int height, char *windowName) {
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(5, 5);
	glutCreateWindow(windowName);
	glutReshapeFunc(reshape_handler);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void display_func(void) {
	char buffer[128];

	// fill draw buffer black
	glClearColor(BLACK.r, BLACK.g, BLACK.b, BLACK.a);
	glClear(GL_COLOR_BUFFER_BIT);

	bpm_draw();

	// flush & swap the buffer
	glFlush();
	glutSwapBuffers();
}

void keyboard_func(uchar_t c, int mouse_x, int mouse_y) {
	bool	trigger_redraw = false;

	switch (c) {
		case 'B':
		case 'b': {
			bpm_reset_pulse_counter();
		} break;
		case ' ': {
			bpm_count_pulse();
			trigger_redraw = true;
		} break;
		case 'Q': // quit the program
		case 'q': {
			exit(0);
		} break;
	}

	if (trigger_redraw)
		glutPostRedisplay();
}

void keyboard_special_func(int c, int mouse_x, int mouse_y) {

}

void mouse_func(int button, int state, int x, int y) {

}

void idle_func(void) {
	bpm_animate_flasher();
	glutPostRedisplay();
}
