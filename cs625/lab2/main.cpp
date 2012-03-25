/******************************************************************************
* CS6250 Lab 2 - Local Reflection Models
* Mark Randles
* 2009-10-05
*
* GOAL: To display a octagon with phong shading.
*
* ARCHITECTURE: Well if it were complete it would load a regular octagon into
*    memory then facet each face into a number of facets that are 3x3x3
*    triangles.  It would then render the octagon coloring each facet by
*    phong shading based on the facet normal.
*
*    It uses OpenGL for rendering, z-buffering, and hidden surface removal.
*
*    This piece of software is not complete.  It cannot render the octagon
*    with shading.
*
* CONTROLS:
*   q/Q: Quit the program
*
* DEBUG MODE: To turn on debug mode toggle the debug define to 1
*****************************************************************************/
#define DEBUG				1
#define WINDOW_X			450
#define WINDOW_Y			450
#define WINDOW_NAME			"CS6250 - Lab #2 - Local Reflection Models"
#define ERROR_MESSAGE_SIZE	1024

#define AMBIENT             1.0
#define DIFFUSE             0.3
#define SPECULAR            0.7
#define SPECULAR_EXP        3

#define DECOMP_VERTEX       9000

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

typedef unsigned char uchar_t;

typedef struct {
	double	x;
	double	y;
	double	z;
} vector_3d_t;

typedef struct {
	vector_3d_t coords;
	vector_3d_t normal;
} vertex_t;

typedef struct {
	int         p1;
	int         p2;
	int         p3;
	vector_3d_t normal;
} facet_t;

typedef struct {
	vertex_t*    vertex;
	int          num_vertex;
	facet_t*     facets;
	int          num_facets;
	vector_3d_t  centroid;
	color_t      color;
} object_t;

typedef struct {
	vertex_t p;
	double   I;
	color_t  color;
	bool     onoff;
} light_t;

void setup(void);
void setupOcta(void);
void setupLights(void);
void reset(void);

void display_func(void);
void keyboard_func(uchar_t, int, int);
void keyboard_special_func(int, int, int);
void mouse_func(int, int, int, int);
void idle_func(void);
void terminate(void);

void difference(vector_3d_t&, vector_3d_t&, vector_3d_t&);
double dot(vector_3d_t&, vector_3d_t&);
void cross(vector_3d_t&, vector_3d_t&, vector_3d_t&);
double length(vector_3d_t&);
void normalize(vector_3d_t&);
void scalar(double, vector_3d_t&);

void facetOcta(void);
int splitTriangle(int, int, int, int, vector_3d_t*);

void phong(vertex_t&, color_t&, color_t&);

void add_error(const char* s, ...);
void clear_error(void);
void print_error(void);

const vector_3d_t COP = {0,0,100}; // center of projection
const vector_3d_t VRP = {0,0,0};   // view reference point
const vector_3d_t VUP = {0,1,0};   // view up
const vector_3d_t VPN = {0,0,1};   // viewport normal

char     error_message[ERROR_MESSAGE_SIZE];
object_t octahedron;
object_t decomp;
int      num_lights;
light_t* lights;

double Ka    = AMBIENT;
double Kd    = DIFFUSE;
double Ks    = SPECULAR;
double alpha = SPECULAR_EXP;

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

void setup(void) {
	clear_error();
	glClearColor(BLACK.r,BLACK.g,BLACK.b,BLACK.a);
	setupOcta();
	facetOcta();
	setupLights();
}

void setupOcta(void) {
	vertex_t    *v;
	facet_t     *f;
	vector_3d_t a,b;

	// clear out the octahedron struct
	memset(&octahedron,0,sizeof(object_t));

	// allocate vertex memory
	octahedron.num_vertex    = 6;
	octahedron.vertex = v = new vertex_t[octahedron.num_vertex];

	// assign vertex
	v->coords.x = -50; v->coords.y = 0;   v->coords.z = 0;   v++; // 0
	v->coords.x = 50;  v->coords.y = 0;   v->coords.z = 0;   v++; // 1
	v->coords.x = 0;   v->coords.y = -50; v->coords.z = 0;   v++; // 2
	v->coords.x = 0;   v->coords.y = 50;  v->coords.z = 0;   v++; // 3
	v->coords.x = 0;   v->coords.y = 0;   v->coords.z = -50; v++; // 4
	v->coords.x = 0;   v->coords.y = 0;   v->coords.z = 50;       // 5

	// allocate facets memory
	octahedron.num_facets = 8;
	octahedron.facets = f = new facet_t[octahedron.num_facets];

	// assign vertex ccw
	f->p1 = 0; f->p2 = 5; f->p3 = 3; f++; // left-top-front
	f->p1 = 0; f->p2 = 2; f->p3 = 5; f++; // left-bottom-front
	f->p1 = 1; f->p2 = 3; f->p3 = 5; f++; // right-bottom-front
	f->p1 = 1; f->p2 = 5; f->p3 = 2; f++; // right-top-front
	f->p1 = 0; f->p2 = 4; f->p3 = 2; f++; // left-top-back
	f->p1 = 0; f->p2 = 3; f->p3 = 4; f++; // left-bottom-back
	f->p1 = 1; f->p2 = 2; f->p3 = 4; f++; // right-bottom-back
	f->p1 = 1; f->p2 = 4; f->p3 = 3;      // right-top-back

	// compute the surface normals
	v = octahedron.vertex;
	for(int i=0; i < octahedron.num_facets; i++) {
		f = &octahedron.facets[i];
		difference(v[f->p2].coords,v[f->p1].coords,a);
		difference(v[f->p3].coords,v[f->p2].coords,b);
		cross(a,b,f->normal);
		normalize(f->normal);
	}

	// set the centroid
	octahedron.centroid.x = 0;
	octahedron.centroid.y = 0;
	octahedron.centroid.z = 0;

	// set the object color
	octahedron.color.r = 1.0;
	octahedron.color.g = 0.0;
	octahedron.color.b = 1.0;
	octahedron.color.a = 1.0;
}

void setupLights(void) {
	light_t *l;

	num_lights = 2;
	lights = new light_t[num_lights];

	// Light #1: (0,0,100) White, full-intensity
	l = &lights[0];
	l->p.coords.x = 0.0;
	l->p.coords.y = 0.0;
	l->p.coords.z = 100.0;
	l->p.normal.x = 0.0;
	l->p.normal.y = 0.0;
	l->p.normal.z = 1.0;
	l->I          = 0.6;
	l->color.r    = WHITE.r * l->I;
	l->color.g    = WHITE.g * l->I;
	l->color.b    = WHITE.b * l->I;
	l->onoff      = true;

	// Light #2: (60,0,0) White, full-intensity
	l = &lights[1];
	l->p.coords.x = 60.0;
	l->p.coords.y = 0.0;
	l->p.coords.z = 0.0;
	l->p.normal.x = 1.0;
	l->p.normal.y = 0.0;
	l->p.normal.z = 0.0;
	l->I          = 0.6;
	l->color.r    = WHITE.r * l->I;
	l->color.g    = WHITE.g * l->I;
	l->color.b    = WHITE.b * l->I;
	l->onoff      = true;

}

void facetOcta(void) {
	vector_3d_t *vertex;
	facet_t     *f;
	int         cnt_vertex=8;

	vertex = new vector_3d_t[DECOMP_VERTEX];
	memset(vertex,0,sizeof(vector_3d_t)*DECOMP_VERTEX);

	for(int i=0; i < octahedron.num_vertex; i++) {
		memcpy(&vertex[i],&octahedron.vertex[i].coords,sizeof(vector_3d_t));
	}
	for(int i=0; i < 1/*octahedron.num_facets*/; i++) {
		f = &octahedron.facets[i];
		cnt_vertex += splitTriangle(f->p1,f->p2,f->p3,cnt_vertex,vertex);
	}
	debug("cnt_vertex = %d",cnt_vertex);

	memset(&decomp,0,sizeof(object_t));

	decomp.num_vertex = cnt_vertex;
	decomp.vertex     = new vertex_t[decomp.num_vertex];
	for(int i=0; i < decomp.num_vertex; i++) {
		memcpy(&decomp.vertex[i].coords,&vertex[i],sizeof(vector_3d_t));

	}

	// set the centroid
	decomp.centroid.x = 0;
	decomp.centroid.y = 0;
	decomp.centroid.z = 0;

	// set the object color
	decomp.color.r = 1.0;
	decomp.color.g = 0.0;
	decomp.color.b = 1.0;
	decomp.color.a = 1.0;
}

int splitTriangle(int p1, int p2, int p3, int i, vector_3d_t *list) {
	vector_3d_t v;
	int         cnt = 3,p1p2m=i,p2p3m=i+1,p3p1m=i+2;

	list[p1p2m].x += (list[p1].x + list[p2].x) / 2;
	list[p1p2m].y += (list[p1].y + list[p2].y) / 2;
	list[p1p2m].z += (list[p1].z + list[p2].z) / 2;
	//debug("p1p2m = (%f,%f,%f)",list[p1p2m].x,list[p1p2m].y,list[p1p2m].z);

	list[p2p3m].x += (list[p2].x + list[p3].x) / 2;
	list[p2p3m].y += (list[p2].y + list[p3].y) / 2;
	list[p2p3m].z += (list[p2].z + list[p3].z) / 2;
	//debug("p2p3m = (%f,%f,%f)",list[p2p3m].x,list[p2p3m].y,list[p2p3m].z);

	list[p3p1m].x += (list[p3].x + list[p1].x) / 2;
	list[p3p1m].y += (list[p3].y + list[p1].y) / 2;
	list[p3p1m].z += (list[p3].z + list[p1].z) / 2;
	//debug("p3p1m = (%f,%f,%f)",list[p3p1m].x,list[p3p1m].y,list[p3p1m].z);

	difference(list[p3p1m],list[p1],v);
	if (length(v) > 4.0) {
		cnt += splitTriangle(p1,p1p2m,p3p1m,i+cnt,list);
		cnt += splitTriangle(p1p2m,p2,p2p3m,i+cnt,list);
		cnt += splitTriangle(p1p2m,p2p3m,p3p1m,i+cnt,list);
		cnt += splitTriangle(p3p1m,p2p3m,p3,i+cnt,list);
	}

	return cnt;
}

void reset(void) {
	clear_error();
	glClearColor(BLACK.r,BLACK.g,BLACK.b,BLACK.a);
}

void phong(vector_3d_t N, color_t& orig_color, color_t &new_color) {
	color_t     ambient, diffuse, specular, light_color;
	vector_3d_t H;
	light_t     *l;

	debug("Ka = %f, Kd = %f, Ks = %f", Ka, Kd, Ks);

	// clear the colors
	memset(&ambient,0,sizeof(color_t));
	memset(&diffuse,0,sizeof(color_t));
	memset(&specular,0,sizeof(color_t));

	// compute the ambient intensity value
	ambient.r = Ka * orig_color.r;
	ambient.g = Ka * orig_color.g;
	ambient.b = Ka * orig_color.b;

	// compute the diffuse intensity value
	for (int i=0; i < num_lights; i++) {
		l          = &lights[i];
		debug("L*R = %f",dot(l->p.normal,N));
		if (l->onoff) {
			debug("diffuse.r = %f",Kd * orig_color.r * dot(l->p.normal,N));
			diffuse.r += Kd * orig_color.r * dot(l->p.normal,N);
			diffuse.g += Kd * orig_color.g * dot(l->p.normal,N);
			diffuse.b += Kd * orig_color.b * dot(l->p.normal,N);
		}
	}

	// compute the specular intensity value
	H.x = (l->p.normal.x + VPN.x) / 2;
	H.y = (l->p.normal.y + VPN.y) / 2;
	H.z = (l->p.normal.z + VPN.z) / 2;
	for (int i=0; i < num_lights; i++) {
		l          = &lights[i];
		if (l->onoff) {
			specular.r += Ks * orig_color.r * pow(dot(l->p.normal,N),alpha);
			specular.g += Ks * orig_color.g * pow(dot(l->p.normal,N),alpha);
			specular.b += Ks * orig_color.b * pow(dot(l->p.normal,N),alpha);
		}
	}

	debug("ambient  = (%f,%f,%f)", ambient.r, ambient.g, ambient.b);
	debug("diffuse  = (%f,%f,%f)", diffuse.r, diffuse.g, diffuse.b);
	debug("specular = (%f,%f,%f)", specular.r, specular.g, specular.b);

	new_color.r = ambient.r + diffuse.r + specular.r;
	new_color.g = ambient.g + diffuse.g + specular.g;
	new_color.b = ambient.b + diffuse.b + specular.b;
	new_color.a = 1.0;
}

void display_func(void) {
	facet_t  *f;
	vertex_t *v1,*v2,*v3;
	color_t   c;

	// clear the buffer
	glClear(GL_COLOR_BUFFER_BIT);

	// set the draw color
	glColor4f(WHITE.r,WHITE.g,WHITE.b,WHITE.a);

	// setup the model view
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// set the camera using COP, VRP, and VUP
	gluLookAt(
		COP.x, COP.y, COP.z,
		VRP.x, VRP.y, VRP.z,
		VUP.x, VUP.y, VUP.z
	);

	// draw triangles
	/*memset(&c,0,sizeof(color_t));
	glBegin(GL_TRIANGLES);
	for (int i=0; i < octahedron.num_facets; i++) {
		f = &octahedron.facets[i];
		debug("Drawing triangle %d:",i);
		debug("f->normal = (%f,%f,%f)",f->normal.x, f->normal.y, f->normal.z);
		phong(f->normal,octahedron.color,c);
		glColor4f(c.r,c.g,c.b,c.a);
		v1 = &octahedron.vertex[f->p1];
		glVertex3d(v1->coords.x,v1->coords.y,v1->coords.z);
		v1 = &octahedron.vertex[f->p2];
		glVertex3d(v1->coords.x,v1->coords.y,v1->coords.z);
		v1 = &octahedron.vertex[f->p3];
		glVertex3d(v1->coords.x,v1->coords.y,v1->coords.z);
		debug("");
	}
	glEnd();
	*/

	// wireframe triangles in white
	/*glBegin(GL_LINES);
	glColor4f(WHITE.r,WHITE.g,WHITE.b,WHITE.a);
	for (int i=0; i < octahedron.num_facets; i++) {
		f  = &octahedron.facets[i];
		v1 = &octahedron.vertex[f->p1];
		v2 = &octahedron.vertex[f->p2];
		v3 = &octahedron.vertex[f->p3];

		glVertex3d(v1->coords.x,v1->coords.y,v1->coords.z);
		glVertex3d(v2->coords.x,v2->coords.y,v2->coords.z);

		glVertex3d(v2->coords.x,v2->coords.y,v2->coords.z);
		glVertex3d(v3->coords.x,v3->coords.y,v3->coords.z);

		glVertex3d(v3->coords.x,v3->coords.y,v3->coords.z);
		glVertex3d(v1->coords.x,v1->coords.y,v1->coords.z);
	}
	glEnd();
	*/

	glBegin(GL_POINTS);
	glColor4f(WHITE.r, WHITE.g, WHITE.b, WHITE.a);
	for (int i=0; i < decomp.num_vertex; i++) {
		glVertex3d(decomp.vertex[i].coords.x,decomp.vertex[i].coords.y,decomp.vertex[i].coords.z);
	}
	glEnd();

	// flush the buffers and swap them
	glFlush();
	glutSwapBuffers();
}

void keyboard_func(uchar_t c, int mouse_x, int mouse_y) {
	int i;

	switch (c) {
		case 'Q': // quit the program
		case 'q': {
			terminate();
			exit(0);
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
	if (octahedron.vertex != NULL) {
		delete [] octahedron.vertex;
		octahedron.vertex = NULL;
	}

	if (octahedron.facets != NULL) {
		delete [] octahedron.facets;
		octahedron.facets = NULL;
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

void difference(vector_3d_t& a, vector_3d_t& b, vector_3d_t& d) {
	d.x = a.x - b.x;
	d.y = a.y - b.y;
	d.z = a.z - b.z;
}

double dot(vector_3d_t& v1, vector_3d_t& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

void cross(vector_3d_t& a, vector_3d_t& b, vector_3d_t& p) {
	p.x = a.y * b.z - a.z * b.y;
	p.y = a.z * b.x - a.x * b.z;
	p.z = a.x * b.y - a.y * b.x;
}

double length(vector_3d_t& v) {
	return sqrt(pow(v.x,2.0) + pow(v.y,2.0) + pow(v.z,2.0));
}

void normalize(vector_3d_t& v) {
	double l = length(v);

	v.x /= l;
	v.y /= l;
	v.z /= l;
}

void scalar(double s, vector_3d_t& v) {
	v.x *= s;
	v.y *= s;
	v.z *= s;
}
