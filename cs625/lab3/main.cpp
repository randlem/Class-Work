/******************************************************************************
* CS6250 Lab 3 - Raycasting a Scene
* Mark Randles
* 2009-10-05
*
* GOAL: To create a raytracer to render a scene.
*
* ARCHITECTURE: The program is broken up into two halves.  One half displays a
*  help menu as text.  The second half is a recursive ray-tracer.  To display
*  the help press the h/H key.  To run the simulation press the s/S key.  To
*  exit the program in either screen press the x/X key.
*
*  The raytracer takes a given screen pixel then parallel projects a scene onto
* that pixel.  Each ray is bounced depth times.  The color for the pixel is the
* collected color after the raytrace has finished.  Each pixel is blitted
* to the screen using opengl calls.
*
*  The size of the screen canvas is set by WINDOW_X and WINDOW_Y.  This allows
* for different qualitites of renders.  For the most part values for the local
* lighting model were determined to make the scene look good.
*
*  The scene contains one white light which is behind the camera.  it can be
* toggled to two different positions using the l/L key.
*
* KNOWN BUGS:
*
*  There are known drawing artifacts, the surface of the objects is "mottled"
* when reflectance is used to color them.  I believe the cause of this is something
* in the sphere intersection algorithm giving false positives.
*
*  No shadows are projected either.  This doesn't create a visible artifact, but
* does restrict the flexiability of the raytracer to render the scene.
*
* CONTROLS:
*   x/X: Quit the program
*   h/H: Display help message
*   s/S: Runs raytrace situation
*   l/L: Move the light position -20
*
* DEBUG MODE: To turn on debug mode toggle the debug define to 1
*****************************************************************************/
#define DEBUG				0
#define WINDOW_X			200
#define WINDOW_Y			200
#define WINDOW_NAME			"CS6250 - Program #3 - Raycasting a Scene"

#define AMBIENT             0.1
#define DIFFUSE             0.2
#define SPECULAR            0.3
#define SPECULAR_EXP        3

#define SCENE_X             100
#define SCENE_Y             100

#define NUM_SPHERES         7
#define MAX_DEPTH           3

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
#include <GL/glu.h>
#include <GL/glut.h>

#include "cs_456_250_setup.h"
#include "util.h"

enum STATE {HELP, RAYTRACE};

typedef struct {
	double x;
	double y;
	double z;
} vector_3d_t;

typedef struct {
	vector_3d_t center;
	uint_t      r;
	color_t     color;
	double      reflectance;
} sphere_t;

typedef struct {
	vector_3d_t position;
	color_t     color;
} light_t;

void setup(int*, char**);
void setup_spheres();
void setup_light();

void display_func(void);
void keyboard_func(uchar_t, int, int);
void keyboard_special_func(int, int, int);
void mouse_func(int, int, int, int);
void idle_func(void);

double dot (const vector_3d_t &, const vector_3d_t &);
void normalize(vector_3d_t &);

double translate_x(int);
double translate_y(int);
void raytrace(vector_3d_t, vector_3d_t, color_t*, int*);
double diffuse(vector_3d_t &, vector_3d_t &);
double specular(vector_3d_t &, vector_3d_t &);
double sphere_intersect(const vector_3d_t &, const vector_3d_t &, const sphere_t &);

void terminate(void);
void teardown_spheres();

void draw_help();

STATE       program = HELP;
sphere_t*   spheres = NULL;
uint_t      curr_light = 0;
light_t     light_source[2];
vector_3d_t VRP     = {50,50,0};

int main(int argc, char* argv[]) {
	// setup the program
	setup(&argc,argv);

	// run the OGL main loop
	glutMainLoop();

	return 0;
}

void setup(int* argc, char** argv) {
	clear_error();

	glutInit(argc,argv);
	init_setup(WINDOW_X,WINDOW_Y,WINDOW_NAME);
	glutDisplayFunc(display_func);
	glutKeyboardFunc(keyboard_func);

	setup_spheres();
	setup_light();

	glClearColor(BLACK.r,BLACK.g,BLACK.b,BLACK.a);
}

void setup_spheres() {
	spheres = new sphere_t[NUM_SPHERES];
	memset(spheres,0,sizeof(sphere_t)*NUM_SPHERES);

	// silver sphere
	spheres[0].center.x    = 0;
	spheres[0].center.y    = 0;
	spheres[0].center.z    = 20;
	spheres[0].r           = 6;
	spheres[0].color.r     = 0.5;
	spheres[0].color.g     = 0.5;
	spheres[0].color.b     = 0.5;
	spheres[0].reflectance = 0.7;

	// red sphere
	spheres[1].center.x    = 15;
	spheres[1].center.y    = 15;
	spheres[1].center.z    = 20;
	spheres[1].r           = 6;
	spheres[1].color.r     = 1.0;
	spheres[1].color.g     = 0.0;
	spheres[1].color.b     = 0.0;
	spheres[1].reflectance = 0.5;

	// green sphere
	spheres[2].center.x    = 78;
	spheres[2].center.y    = 52;
	spheres[2].center.z    = 70;
	spheres[2].r           = 7;
	spheres[2].color.r     = 0.0;
	spheres[2].color.g     = 1.0;
	spheres[2].color.b     = 0.0;
	spheres[2].reflectance = 0.5;

	// blue sphere
	spheres[3].center.x    = 48;
	spheres[3].center.y    = 51;
	spheres[3].center.z    = 68;
	spheres[3].r           = 8;
	spheres[3].color.r     = 0.0;
	spheres[3].color.g     = 0.0;
	spheres[3].color.b     = 1.0;
	spheres[3].reflectance = 0.5;

	// yellow sphere
	spheres[4].center.x    = 50;
	spheres[4].center.y    = 50;
	spheres[4].center.z    = 40;
	spheres[4].r           = 4;
	spheres[4].color.r     = 1.0;
	spheres[4].color.g     = 1.0;
	spheres[4].color.b     = 0.0;
	spheres[4].reflectance = 0.5;

	// orange sphere
	spheres[5].center.x    = -9;
	spheres[5].center.y    = 11;
	spheres[5].center.z    = 11;
	spheres[5].r           = 10;
	spheres[5].color.r     = 1.0;
	spheres[5].color.g     = 0.57;
	spheres[5].color.b     = 0.20;
	spheres[5].reflectance = 0.5;

	// white sphere
	spheres[6].center.x    = 3;
	spheres[6].center.y    = 11;
	spheres[6].center.z    = 11;
	spheres[6].r           = 2;
	spheres[6].color.r     = 1.0;
	spheres[6].color.g     = 1.0;
	spheres[6].color.b     = 1.0;
	spheres[6].reflectance = 0.5;
}

void setup_light() {
	light_source[0].position.x = 50;
	light_source[0].position.y = 50;
	light_source[0].position.z = -10;

	light_source[0].color.r = 1.0;
	light_source[0].color.g = 1.0;
	light_source[0].color.b = 1.0;

	light_source[1].position.x = 30;
	light_source[1].position.y = 50;
	light_source[1].position.z = -10;

	light_source[1].color.r = 1.0;
	light_source[1].color.g = 1.0;
	light_source[1].color.b = 1.0;
}

void display_func(void) {
	color_t c;
	vector_3d_t r0,rd;
	int depth;

	glClear(GL_COLOR_BUFFER_BIT);

	switch (program) {
		case RAYTRACE: {
			for (int h=WINDOW_Y; h >= 0; h--) {
				for (int w=0; w <= WINDOW_X; w++) {
					r0.x = translate_x(w);
					r0.y = translate_y(h);
					r0.z = 0.0;

					rd.x = 0.0;
					rd.y = 0.0;
					rd.z = 1.0;
					normalize(rd);

					depth = 3;

					raytrace(r0,rd,&c,&depth);

					glColor3f(c.r,c.g,c.b);
					glBegin(GL_POINTS);
						glVertex2i(w,h);
					glEnd();
				}
			}

		} break;
		case HELP:
		default: {
			draw_help();
		} break;
	}

	glFlush();
	glutSwapBuffers();
}

void keyboard_func(uchar_t c, int mouse_x, int mouse_y) {
	switch (c) {
		case 'S':
		case 's': {
			program = RAYTRACE;
		} break;
		case 'H':
		case 'h': {
			program = HELP;
		} break;
		case 'L':
		case 'l': {
			curr_light = (curr_light + 1) % 2;
			debug("curr_light = %d",curr_light);
		} break;
		case 'X': // quit the program
		case 'x': {
			terminate();
			exit(0);
		} break;
	}

	glutPostRedisplay();
}

double dot (const vector_3d_t &v0, const vector_3d_t &v1) {
	return (v0.x * v1.x) + (v0.y * v1.y) + (v0.z * v1.z);
}

void normalize(vector_3d_t& v) {
	double l = sqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z));

	v.x /= l;
	v.y /= l;
	v.z /= l;
}

double translate_x(int x) {
	return (x * (SCENE_X / (double)WINDOW_X));// - (SCENE_X / 2.0);
}

double translate_y(int y) {
	return (y * (SCENE_Y / (double)WINDOW_Y));// - (SCENE_Y / 2.0);
}

void raytrace(vector_3d_t r0, vector_3d_t rd, color_t* color, int* depth) {
	double t,min_t,d,local_co;
	int t_i;
	vector_3d_t ri,rf,norm,light,viewer;
	sphere_t *s;
	bool shadowed;
	color_t nc;

	if (*depth == 0)
		return;

	// test for sphere intersection
	t_i=-1; min_t = 100000000.0;
	for (int i=0; i < NUM_SPHERES; i++) {
		t = sphere_intersect(r0,rd,spheres[i]);

		if (t > 0.0 && t < min_t) {
			//debug("%d: <%f,%f,%f> <%f,%f,%f> %f",*depth,r0.x,r0.y,r0.z,rd.x,rd.y,rd.z,t);
			min_t = t;
			t_i   = i;
		}
	}
	//debug("%f %d",min_t,t_i);

	// if sphere intersected, compute the color
	if (t_i >= 0) {
		s = &spheres[t_i];

		ri.x = r0.x + rd.x * min_t;
		ri.y = r0.y + rd.y * min_t;
		ri.z = r0.z + rd.z * min_t;

		norm.x = -(s->center.x - ri.x) / s->r;
		norm.y = -(s->center.y - ri.y) / s->r;
		norm.z = -(s->center.z - ri.z) / s->r;

		d    = dot(norm,rd);
		rf.x = rd.x - (2.0 * norm.x * d);
		rf.y = rd.y - (2.0 * norm.y * d);
		rf.z = rd.z - (2.0 * norm.z * d);
		normalize(rf);
		//debug("rf = <%f,%f,%f>",rf.x,rf.y,rf.z);

		(*depth)--;
		raytrace(ri,rf,&nc,depth);

		light.x = light_source[curr_light].position.x - ri.x;
		light.y = light_source[curr_light].position.y - ri.y;
		light.z = light_source[curr_light].position.z - ri.z;
		normalize(light);

		// see if the point is in shadow
//		for (int i=0; i < NUM_SPHERES; i++) {
//			t = sphere_intersect(ri,light,spheres[i]);

//			if (t >	 0.0) {
//				color->r = (s->color.r * AMBIENT) + (s->reflectance * nc.r);
//				color->g = (s->color.g * AMBIENT) + (s->reflectance * nc.g);
//				color->b = (s->color.b * AMBIENT) + (s->reflectance * nc.b);
//			    return;
//			}
//		}

		viewer.x = VRP.x - ri.x;
		viewer.y = VRP.y - ri.y;
		viewer.z = VRP.z - ri.z;
		normalize(viewer);

		// new   = base color * (AMBIENT + diffuse()           + specular()    ) + reflected color
		local_co = diffuse(norm,light) + specular(rf,viewer);
		color->r = (s->color.r * AMBIENT) + (light_source[curr_light].color.r * local_co) + (s->reflectance * nc.r);
		color->g = (s->color.g * AMBIENT) + (light_source[curr_light].color.g * local_co) + (s->reflectance * nc.g);
		color->b = (s->color.b * AMBIENT) + (light_source[curr_light].color.b * local_co) + (s->reflectance * nc.b);
		return;
	}

	// test for wall intersections

	// if wall intersected, compute the color

	color->r = 0.1;
	color->g = 0.1;
	color->b = 0.1;
}

double diffuse(vector_3d_t & N, vector_3d_t & L) {
	double d = DIFFUSE * dot(L,N);
	return (d <= 0.0) ? 0.0 : d;
}

double specular(vector_3d_t & R, vector_3d_t & V) {
	double s = SPECULAR * pow(dot(R,V),SPECULAR_EXP);
	return (s <= 0.0) ? 0.0 : s;
}

double sphere_intersect(const vector_3d_t &r0, const vector_3d_t &rd, const sphere_t &s) {
	double B,C,disc,t;

	B = 2 * (rd.x * (r0.x - s.center.x) +
	         rd.y * (r0.y - s.center.y) +
	         rd.z * (r0.z - s.center.z));
	C = ((r0.x - s.center.x) * (r0.x - s.center.x)) +
	    ((r0.y - s.center.y) * (r0.y - s.center.y)) +
	    ((r0.z - s.center.z) * (r0.z - s.center.z)) - (s.r * s.r);

	disc = (B * B) - (4 * C);

	if (disc < 0.0)
		return disc;

	t = (-B - sqrt(disc)) / 2.0;

	if (t > 0.0)
		return t;

	t = (-B + sqrt(disc)) / 2.0;

	return t;
}

void terminate(void) {
	teardown_spheres();
}

void teardown_spheres() {
	delete [] spheres;
	spheres = NULL;
}

void draw_help() {
	print_string(33,3,GLUT_BITMAP_HELVETICA_12,ORANGE,"Press H to display this help");
	print_string(18,3,GLUT_BITMAP_HELVETICA_12,ORANGE,"Press S to begin simulation");
	print_string(3,3,GLUT_BITMAP_HELVETICA_12,ORANGE,"Press X to exit program");
}
