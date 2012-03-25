/******************************************************************************
* CS6250 Lab 1 - Wireframe Projections
* Mark Randles
* 2009-09-21
*
* GOAL: To load an arbitrary object file in a known file format from the disk
* and then render the object in a window based either using a parallel or
* perspective projection.  The object can be rotated about the y-axis centroid
* and translated in the x and y plane.  Files are loaded after a filename is
* input in the program.
*
* ARCHITECTURE: The main program is designed as a finite-state machine.  The
* two program states are "normal" and "filename input".  There are also two
* drawing states, "parallel" and "perspective", each of which uses that
* projection method to draw the object.
*
* The primary window is redrawn with every keyboard press, but is otherwise
* static, since there is no animation.  Different things are drawn based on
* what state the program is in.
*
* CONTROLS:
* 	'q' or 'Q' -- Quit the program
* 	'x' or 'X' -- Translate the object by +/-10 units along the x axis
*	'y' or 'Y' -- Translate the object by +/-10 units along the y axis
*	'z' or 'Z' -- Translate the object by +/-10 units along the z axis
*	'r'        -- Rotate the object about the y axis by approx 30 deg
*	'p'        -- Toggle between parallel/perspective projection
* 	'I'		   -- Prompt for inputting a filename
*	While the prompt for inputting the filename is up, all keyboard input is
*		captured as part of the filename.  Press enter when the filename is
*		complete, to load and render the file.  Backspace is supported.
*
* DEBUG MODE: To turn on debug mode toggle the debug define to 1
*****************************************************************************/
#define DEBUG				0
#define WINDOW_X			450
#define WINDOW_Y			450
#define WINDOW_NAME			"CS6250 -- Lab 1"
#define FILENAME_SIZE		30
#define ERROR_MESSAGE_SIZE	1024

#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ifstream;

#include <string>
using std::string;

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <GL/gl.h>
#include <GL/glut.h>

#include "cs_456_250_setup.h"
#include "util.h"

typedef unsigned char uchar_t;

typedef struct {
	int		x;
	int		y;
} point_t;

typedef struct {
	float	x;
	float	y;
	float	z;
} vertex_t;

typedef struct {
	int		p1;
	int		p2;
} line_t;

typedef struct {
	int			vertex_cnt;
	vertex_t*	vertex;
	int			line_cnt;
	line_t*		lines;
	vertex_t	centroid;
} object_t;

enum STATE {NORMAL, FILENAME_INPUT};
enum PROJECTION_METHOD {PARALLEL,PERSPECTIVE};

void setup(void);
void reset(void);
void load_file(char*);
void display_func(void);
void keyboard_func(uchar_t, int, int);
void keyboard_special_func(int, int, int);
void mouse_func(int, int, int, int);
void idle_func(void);
void terminate(void);

void add_error(const char* s, ...);
void clear_error(void);
void print_error(void);

void projection_parallel(const vertex_t*, point_t*);
void projection_perspective(const vertex_t*, point_t*);
void project_point(vertex_t*, point_t*);
void translate_x(int);
void translate_y(int);
void rotate_y(float);
void compute_centroid(vertex_t*);

void draw_line(point_t*, point_t*, const color_t&);

const vertex_t COP = {0,0,100}; 			// center of projection
const vertex_t VRP = {0,0,0};				// view reference point
const vertex_t VUP = {0,1,0};				// view up
const vertex_t VPN = {0,0,1};				// viewport normal

char error_message[ERROR_MESSAGE_SIZE];
STATE program_state = NORMAL;				// current program state
char filename[FILENAME_SIZE];				// currently loaded file
object_t object = {0, NULL, 0, NULL};		// currently loaded object
PROJECTION_METHOD projection_method = PARALLEL;

int main(int argc, char* argv[]) {
	// setup the program
	setup();
	glutInit(&argc,argv);
	init_setup(WINDOW_X,WINDOW_Y,WINDOW_NAME);
	glutDisplayFunc(display_func);
	glutKeyboardFunc(keyboard_func);
	glutSpecialFunc(keyboard_special_func);
	glutMouseFunc(mouse_func);
	glutIdleFunc(idle_func);

	// run the OGL main loop
	glutMainLoop();

	return 0;
}

void projection_parallel(const vertex_t* v, point_t* p) {
	p->x = (int)v->x + (WINDOW_X / 2);
	p->y = (int)v->y + (WINDOW_Y / 2);
}

void projection_perspective(const vertex_t* v, point_t* p) {
	p->x = (int)(v->x * COP.z / (COP.z - v->z)) + (WINDOW_X / 2);
	p->y = (int)(v->y * COP.z / (COP.z - v->z)) + (WINDOW_Y / 2);
}

void project_point(vertex_t* v, point_t* p) {
	switch (projection_method) {
		case PARALLEL: {
			projection_parallel(v,p);
		} break;
		case PERSPECTIVE: {
			projection_perspective(v,p);
		} break;
	}
}

void translate_x(int dx) {
	for (int i=0; i < object.vertex_cnt; i++)
		object.vertex[i].x += dx;
	compute_centroid(&object.centroid);
}

void translate_y(int dy) {
	for (int i=0; i < object.vertex_cnt; i++)
		object.vertex[i].y += dy;
	compute_centroid(&object.centroid);
}

void translate_z(int dz) {
	for (int i=0; i < object.vertex_cnt; i++)
		object.vertex[i].z += dz;
	compute_centroid(&object.centroid);
}

void rotate_y(float theta) {
	vertex_t *v;
	float x, z, offset_x, offset_z;

	// pre-calc the translation offset for the center of the object
	offset_x = object.centroid.x * cos(theta) + object.centroid.z * sin(theta) - object.centroid.x;
	offset_z = -1.0 * object.centroid.x * sin(theta) + object.centroid.z * cos(theta) - object.centroid.z;

	for (int i=0; i < object.vertex_cnt; i++) {
		v = &object.vertex[i];
		x = v->x;
		z = v->z;
		v->x = x * cos(theta) + z * sin(theta) + offset_x;
		v->z = -1.0 * x * sin(theta) + z * cos(theta) + offset_z;
	}
}

// compute the centroid by averaging the object points
void compute_centroid(vertex_t* c) {
	memcpy(c,&object.vertex[0],sizeof(vertex_t));
	for (int i=1; i < object.vertex_cnt; i++) {
		c->x += object.vertex[i].x;
		c->y += object.vertex[i].y;
		c->z += object.vertex[i].z;
	}
	c->x /= object.vertex_cnt;
	c->y /= object.vertex_cnt;
	c->z /= object.vertex_cnt;
	debug ("centroid = <%0.2f,%0.2f,%0.2f>",c->x,c->y,c->z);
}

void setup(void) {
	clear_error();
	object.vertex = NULL;
	object.lines = NULL;
}

void reset(void) {
	clear_error();

	delete [] object.vertex;
	object.vertex = NULL;
	object.vertex_cnt = 0;

	delete [] object.lines;
	object.lines = NULL;
	object.line_cnt = 0;
}

// Load the file using the following format:
//     First line: <vertex cnt> <line cnt>
//	   Next <vertex cnt> lines: <x,y,z>
//     Next <line cnt> lines: <start point> <end point>
void load_file(char* filename, object_t* obj) {
	ifstream in;
	int i;

	in.open(filename,ifstream::in);
	if (!in.is_open()) {
		debug("Failed to open file %s",filename);
		add_error("Unable to open file %s!", filename);
		return;
	}
	debug("Succeeded in opening file %s",filename);

	in >> obj->vertex_cnt >> obj->line_cnt;

	debug("Loading %d vertex and %d lines",
		obj->vertex_cnt, obj->line_cnt);

	obj->vertex = new vertex_t[obj->vertex_cnt];
	for (i=0; i < obj->vertex_cnt; i++)
		in >> obj->vertex[i].x >> obj->vertex[i].y >> obj->vertex[i].z;

	obj->lines = new line_t[obj->line_cnt];
	for (i=0; i < obj->line_cnt; i++)
		in >> obj->lines[i].p1 >> obj->lines[i].p2;

	in.close();

	compute_centroid(&object.centroid);
}

void display_func(void) {
	line_t* l;
	point_t** translated;

	// blank the background
	glClearColor(BLACK.r, BLACK.g, BLACK.b, BLACK.a);
	glClear(GL_COLOR_BUFFER_BIT);

	switch (program_state) {
		case FILENAME_INPUT: {
			print_string(WINDOW_Y/2, 3, GLUT_BITMAP_9_BY_15, ORANGE,
				"Enter filename : %s", filename);
		} break;
		case NORMAL:
		default: {
			if (object.vertex_cnt <= 1)
				break;

			translated = new point_t*[object.vertex_cnt];
			memset(translated,0,sizeof(point_t*)*object.vertex_cnt);
			for (int i=0; i < object.line_cnt; i++) {
				l = &object.lines[i];
				if (translated[l->p1] == NULL) {
					translated[l->p1] = new point_t;
					project_point(&object.vertex[l->p1],translated[l->p1]);
				}
				if (translated[l->p2] == NULL) {
					translated[l->p2] = new point_t;
					project_point(&object.vertex[l->p2],translated[l->p2]);
				}
				draw_line(translated[l->p1],translated[l->p2],GREEN);
			}

			print_string(WINDOW_Y-15, 3, GLUT_BITMAP_9_BY_15, ORANGE,
				"%s", filename);
		} break;
	}

	if (strlen(error_message) > 0)
		print_error();

	glFlush();
	glutSwapBuffers();
}

void draw_line(point_t* p1, point_t* p2, const color_t &c) {
	glColor4f(c.r, c.g, c.b, c.a);
	glBegin(GL_LINES);
		glVertex2i(p1->x,p1->y);
		glVertex2i(p2->x,p2->y);
	glEnd();
}

void keyboard_func(uchar_t c, int mouse_x, int mouse_y) {
	int i;

	switch (program_state) {
		case FILENAME_INPUT: {
			i = strlen(filename);
			if (c == '\n' || c == '\r') {
				debug("Trying to load file %s ...", filename);
				reset();
				load_file(filename,&object);
				program_state = NORMAL;
				break;
			}
			else if (c == 8) {
				i--;
				c = '\0';
			}
			filename[i] = c;
		} break;
		case NORMAL:
		default: {
			switch (c) {
				case 'x': {
					translate_x(10);
				} break;
				case 'X': {
					translate_x(-10);
				} break;
				case 'y': {
					translate_y(10);
				} break;
				case 'Y': {
					translate_y(-10);
				} break;
				case 'z': {
					translate_z(10);
				} break;
				case 'Z': {
					translate_z(-10);
				} break;
				case 'r': {
					rotate_y(0.5236);
				} break;
				case 'p': {
					if (projection_method == PARALLEL)
						projection_method = PERSPECTIVE;
					else
						projection_method = PARALLEL;
				} break;
				case 'I': {
					memset(filename,'\0',sizeof(char)*FILENAME_SIZE);
					program_state = FILENAME_INPUT;
				} break;
				case 'Q': // quit the program
				case 'q': {
					terminate();
					exit(0);
				} break;
			}
		} break;
	}

	glutPostRedisplay();
}

void keyboard_special_func(int c, int mouse_x, int mouse_y) {

}

void mouse_func(int button, int state, int x, int y) {

}

void idle_func(void) {

}

void terminate(void) {
	if (object.vertex != NULL) {
		delete [] object.vertex;
		object.vertex = NULL;
	}

	if (object.lines != NULL) {
		delete [] object.lines;
		object.lines = NULL;
	}
}

void add_error(const char* s, ...) {
	va_list args;
	va_start(args, s);
	vsprintf(error_message, s, args);
	va_end(args);
}

void clear_error(void) {
	memset(error_message,'\0',sizeof(char)*ERROR_MESSAGE_SIZE);
}

void print_error(void) {
	debug("%s",error_message);
	print_string(3,3, GLUT_BITMAP_9_BY_15, RED, "%s", error_message);
	clear_error();
}

