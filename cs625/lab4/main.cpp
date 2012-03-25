/******************************************************************************
* CS6250 Programming Assignment #4 - Surface Fitting and Modeling
* Mark Randles
* 2009-12-14
*
* GOAL: To explore the rendering of Bezier patches using OpenGL and GLUT.
*
* ARCHITECTURE: A 3 dimensional set of control points is defined which is
*   decomposed into a set of Bezier patches which are rendered as a set of
*   GL_QUADs.  Recomputation is avoided when possible.  THe outline of the
*   room is rendered in green wireframe.
*
* KNOWN BUGS:
*   - Sometimes the room outline is drawn over the Bezier surface.  This is
*     an artifact of OpenGL depth buffering
*   - No shading.
*
* ADDITIONAL FEATURES:
*   - Camera control: using the keyboard the camera can be moved about the
*     scene to look at it from other angles
*   - Different drawing types (wireframe, cloud of points, solid surface)
*
* CONTROLS:
*   q/Q: Quit the program
*   w/a/s/d: move the eye point up/left/down/right
*   t/f/g/h: move the view reference point up/left/down/right
*   r/R: reset the eye and view reference point
*   z/Z: move in/out along the z axis.
*   b/B: change the drawing type (wireframe, cloud of points, solid surface)
*
* DEBUG MODE: To turn on debug mode toggle the debug define to 1
*****************************************************************************/
#define DEBUG				1
#define WINDOW_X			600
#define WINDOW_Y			600
#define WINDOW_NAME			"CS6250 - Program #4 - Surface Fitting and Modeling"

#define PITCH               0.05

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

typedef struct {
	float x;
	float y;
	float z;
} vector_3d_t;

typedef struct {
	matrix_t x;
	matrix_t y;
	matrix_t z;
} bezier_patch_t;

enum DRAW_TYPE { POINTS, WIREFRAME, SOLID };

void setup(int*, char**);
void setup_bezier();
void setup_curtain();
void decomp_bezier();

void display_func(void);
void keyboard_func(uchar_t, int, int);

void terminate(void);
void terminate_bezier();

void draw_room();
void draw_curtain();
void draw_bezier(bezier_patch_t*, DRAW_TYPE);

vector_3d_t eye = {50.0, 50.0, 121.0};
vector_3d_t vrp = {50.0, 50.0, -50.0};
vector_3d_t vup = {0.0, 0.0, -1.0};

matrix_t ctrl_pts[3];
matrix_t blending_func;
int patches_cnt;
bezier_patch_t* patches;
DRAW_TYPE curr_draw_type = SOLID;

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

	glClearColor(BLACK.r,BLACK.g,BLACK.b,BLACK.a);

	setup_bezier();
	setup_curtain();
}

void setup_bezier() {
	double* row;

	// create the blending function
	matrix_allocate(&blending_func,4,4);
	row = blending_func.cells[0];
	row[0] = -1; row[1] = 3; row[2] = -3; row[3] = 1;
	row = blending_func.cells[1];
	row[0] = 3; row[1] = -6; row[2] = 3; row[3] = 0;
	row = blending_func.cells[2];
	row[0] = -3; row[1] = 3; row[2] = 0; row[3] = 0;
	row = blending_func.cells[3];
	row[0] = 1; row[1] = 0; row[2] = 0; row[3] = 0;
}

void setup_curtain() {
	double* r;
	matrix_t t;

	matrix_allocate(&ctrl_pts[0],4,7);
	matrix_allocate(&ctrl_pts[1],4,7);
	matrix_allocate(&ctrl_pts[2],4,7);

	// x-coord ctrl points
	r = ctrl_pts[0].cells[0];
	r[0] = 85.0; r[1] = 70.0; r[2] = 100.0; r[3] = 85.0; r[4] = 70.0; r[5] = 100.0; r[6] = 85.0;
	r = ctrl_pts[0].cells[1];
	r[0] = 85.0; r[1] = 70.0; r[2] = 100.0; r[3] = 85.0; r[4] = 70.0; r[5] = 100.0; r[6] = 85.0;
	r = ctrl_pts[0].cells[2];
	r[0] = 85.0; r[1] = 70.0; r[2] = 100.0; r[3] = 85.0; r[4] = 70.0; r[5] = 100.0; r[6] = 85.0;
	r = ctrl_pts[0].cells[3];
	r[0] = 85.0; r[1] = 70.0; r[2] = 100.0; r[3] = 85.0; r[4] = 70.0; r[5] = 100.0; r[6] = 85.0;

	// y-coord ctrl points
	r = ctrl_pts[1].cells[0];
	r[0] = 0.0; r[1] = 0.0; r[2] = 0.0; r[3] = 0.0; r[4] = 0.0; r[5] = 0.0; r[6] = 0.0;
	r = ctrl_pts[1].cells[1];
	r[0] = 33.3; r[1] = 33.3; r[2] = 33.3; r[3] = 33.3; r[4] = 33.3; r[5] = 33.3; r[6] = 33.3;
	r = ctrl_pts[1].cells[2];
	r[0] = 66.7; r[1] = 66.7; r[2] = 66.7; r[3] = 66.7; r[4] = 66.7; r[5] = 66.7; r[6] = 66.7;
	r = ctrl_pts[1].cells[3];
	r[0] = 100.0; r[1] = 100.0; r[2] = 100.0; r[3] = 100.0; r[4] = 100.0; r[5] = 100.0; r[6] = 100.0;

	// z-coord ctrl points
	r = ctrl_pts[2].cells[0];
	r[0] = 0; r[1] = -16.7; r[2] = -33.3; r[3] = -50; r[4] = -66.7; r[5] = -83.3; r[6] = -100.0;
	r = ctrl_pts[2].cells[1];
	r[0] = 0; r[1] = -16.7; r[2] = -33.3; r[3] = -50; r[4] = -66.7; r[5] = -83.3; r[6] = -100.0;
	r = ctrl_pts[2].cells[2];
	r[0] = 0; r[1] = -16.7; r[2] = -33.3; r[3] = -50; r[4] = -66.7; r[5] = -83.3; r[6] = -100.0;
	r = ctrl_pts[2].cells[3];
	r[0] = 0; r[1] = -16.7; r[2] = -33.3; r[3] = -50; r[4] = -66.7; r[5] = -83.3; r[6] = -100.0;

	decomp_bezier();
}

void decomp_bezier() {
	matrix_t Bt,t,cp[3];
	int i,j,k;

	// generate the transpose of the blending func
	matrix_allocate(&Bt,blending_func.rows,blending_func.cols);
	matrix_transpose(&blending_func,&Bt);

	// allocate a temp matrix
	matrix_allocate(&t,4,4);
	matrix_allocate(&cp[0],4,4);
	matrix_allocate(&cp[1],4,4);
	matrix_allocate(&cp[2],4,4);

	// calc the number of patches from the ctrl points
	patches_cnt = ((int)((ctrl_pts[0].cols - 4.0f) / 3.0f)) + 1;
	debug("patches = %d",patches_cnt);

	// alloate the patches
	patches = new bezier_patch_t[patches_cnt];

	// create the individual patches
	for(i=0; i < patches_cnt; i++) {
		// copy out the control point values
		for(j=0; j < 4; j++) {
			for(k=0; k < 4; k++) {
				cp[0].cells[j][k] = ctrl_pts[0].cells[j][k+(i*3)];
				cp[1].cells[j][k] = ctrl_pts[1].cells[j][k+(i*3)];
				cp[2].cells[j][k] = ctrl_pts[2].cells[j][k+(i*3)];
			}
		}

		// allocate some memory
		matrix_allocate(&patches[i].x,4,4);
		matrix_allocate(&patches[i].y,4,4);
		matrix_allocate(&patches[i].z,4,4);

		// compute the final blending functions
		matrix_multiply(&cp[0],&Bt,&t);
		matrix_multiply(&blending_func,&t,&patches[i].x);
		matrix_multiply(&cp[1],&Bt,&t);
		matrix_multiply(&blending_func,&t,&patches[i].y);
		matrix_multiply(&cp[2],&Bt,&t);
		matrix_multiply(&blending_func,&t,&patches[i].z);
	}

}

void display_func(void) {
	GLfloat light_position[] = { 50.0, 50.0, 121.0, 0.0 };

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//glTranslatef(eye.x,eye.y,eye.z);
	gluLookAt(eye.x,eye.y,eye.z,
			  vrp.x,vrp.y,vrp.z,
			  0.0,1.0,0.0);

//	glColor3f(GREEN.r,GREEN.g,GREEN.b);
//	glPushMatrix();
//	glTranslatef(2.0,2.0,2.0);
//	glRotatef(45.0,0,1.0,1.0);
//	glutWireTeapot(1.0);
//	glPopMatrix();

	draw_room();
	draw_curtain();

	glFlush();
	glutSwapBuffers();
}

void keyboard_func(uchar_t c, int mouse_x, int mouse_y) {
	switch (c) {
		case 's': {
			eye.y -= 1.0f;
		} break;
		case 'w': {
			eye.y += 1.0f;
		} break;
		case 'a': {
			eye.x -= 1.0f;
		} break;
		case 'd': {
			eye.x += 1.0f;
		} break;
		case 'z': {
			eye.z -= 1.0f;
		} break;
		case 'x': {
			eye.z += 1.0f;
		} break;
		case 'f': {
			vrp.x -= 1.0f;
		} break;
		case 'h': {
			vrp.x += 1.0f;
		} break;
		case 't': {
			vrp.y += 1.0f;
		} break;
		case 'g': {
			vrp.y -= 1.0f;
		} break;
		case 'r':
		case 'R': {
			eye.x = 50.0; eye.y = 50.0; eye.z = 121.0;
			vrp.x = 50.0; vrp.y = 50.0; vrp.z = -50.0;
		} break;
		case 'b':
		case 'B': {
			curr_draw_type = (DRAW_TYPE)((curr_draw_type + 1) % 3);
		} break;
		case 'Q': // quit the program
		case 'q': {
			terminate();
			exit(0);
		} break;
	}

	glutPostRedisplay();
}

void terminate(void) {

}

void draw_curtain() {
	int i;

	glColor3f(RED.r,RED.g,RED.b);
	for(i=0; i < patches_cnt; i++) {
		draw_bezier(&patches[i],curr_draw_type);
	}
}

void draw_bezier(bezier_patch_t* patch, DRAW_TYPE draw_type) {
	int i;
	float s,t,s2,t2,s3,t3,x,y,z;
	double **Bx,**By,**Bz;
	vector_3d_t curr_row[21], prev_row[21];

	// grab the blending funcs
	Bx = patch->x.cells;
	By = patch->y.cells;
	Bz = patch->z.cells;

	memset(curr_row,0,sizeof(vector_3d_t)*21);
	memset(prev_row,0,sizeof(vector_3d_t)*21);

	// fill the prev row with s=0.0
	i = 0;
	for(t=0.0; t < 1.0f+PITCH; t+=PITCH) {
		t2 = t * t;
		t3 = t2 * t;
		prev_row[i].x = (Bx[3][0] * t3 + Bx[3][1] * t2 + Bx[3][2] * t + Bx[3][3]);
		prev_row[i].y = (By[3][0] * t3 + By[3][1] * t2 + By[3][2] * t + By[3][3]);
		prev_row[i].z = (Bz[3][0] * t3 + Bz[3][1] * t2 + Bz[3][2] * t + Bz[3][3]);
		i++;
	}

	for(s=PITCH; s < 1.0f+PITCH; s+=PITCH) {
		s2 = s * s;
		s3 = s2 * s;
		i  = 1;
		curr_row[0].x = s3 * Bx[0][3] + s2 * Bx[1][3] + s * Bx[2][3] + Bx[3][3];
		curr_row[0].y = s3 * By[0][3] + s2 * By[1][3] + s * By[2][3] + By[3][3];
		curr_row[0].z = s3 * Bz[0][3] + s2 * Bz[1][3] + s * Bz[2][3] + Bz[3][3];
		for(t=PITCH; t < 1.0f+PITCH; t+=PITCH) {
			t2 = t * t;
			t3 = t2 * t;

			curr_row[i].x = s3 * (Bx[0][0] * t3 + Bx[0][1] * t2 + Bx[0][2] * t + Bx[0][3]) +
							s2 * (Bx[1][0] * t3 + Bx[1][1] * t2 + Bx[1][2] * t + Bx[1][3]) +
							s  * (Bx[2][0] * t3 + Bx[2][1] * t2 + Bx[2][2] * t + Bx[2][3]) +
								 (Bx[3][0] * t3 + Bx[3][1] * t2 + Bx[3][2] * t + Bx[3][3]);

			curr_row[i].y = s3 * (By[0][0] * t3 + By[0][1] * t2 + By[0][2] * t + By[0][3]) +
							s2 * (By[1][0] * t3 + By[1][1] * t2 + By[1][2] * t + By[1][3]) +
							s  * (By[2][0] * t3 + By[2][1] * t2 + By[2][2] * t + By[2][3]) +
								 (By[3][0] * t3 + By[3][1] * t2 + By[3][2] * t + By[3][3]);

			curr_row[i].z = s3 * (Bz[0][0] * t3 + Bz[0][1] * t2 + Bz[0][2] * t + Bz[0][3]) +
							s2 * (Bz[1][0] * t3 + Bz[1][1] * t2 + Bz[1][2] * t + Bz[1][3]) +
							s  * (Bz[2][0] * t3 + Bz[2][1] * t2 + Bz[2][2] * t + Bz[2][3]) +
								 (Bz[3][0] * t3 + Bz[3][1] * t2 + Bz[3][2] * t + Bz[3][3]);

			switch (draw_type) {
				case POINTS: {
					glBegin(GL_POINTS);
						glVertex3f(curr_row[i].x,curr_row[i].y,curr_row[i].z);
					glEnd();
				} break;
				case WIREFRAME: {
					glBegin(GL_LINE_LOOP);
						glVertex3f(prev_row[i].x,prev_row[i].y,prev_row[i].z);
						glVertex3f(prev_row[i-1].x,prev_row[i-1].y,prev_row[i-1].z);
						glVertex3f(curr_row[i-1].x,curr_row[i-1].y,curr_row[i-1].z);
						glVertex3f(curr_row[i].x,curr_row[i].y,curr_row[i].z);
					glEnd();
				} break;
				case SOLID: {
					glBegin(GL_QUADS);
						glVertex3f(prev_row[i].x,prev_row[i].y,prev_row[i].z);
						glVertex3f(prev_row[i-1].x,prev_row[i-1].y,prev_row[i-1].z);
						glVertex3f(curr_row[i-1].x,curr_row[i-1].y,curr_row[i-1].z);
						glVertex3f(curr_row[i].x,curr_row[i].y,curr_row[i].z);
					glEnd();
				}
			}

			i++;
		}

		memcpy(prev_row,curr_row,sizeof(vector_3d_t)*21);
	}

}

void draw_room() {
	glColor3f(GREEN.r,GREEN.g,GREEN.b);
	glBegin(GL_LINE_LOOP);
		glVertex3f(0.0,0.0,0.0);      // front-left-bottom
		glVertex3f(100.0,0.0,0.0);    // front-right-bottom
		glVertex3f(100.0,0.0,-100.0); // back-right-bottom
		glVertex3f(0.0,0.0,-100.0);   // back-left-bottom
	glEnd();
	glBegin(GL_LINE_LOOP);
		glVertex3f(0.0,100.0,0.0);      // front-left-top
		glVertex3f(100.0,100.0,0.0);    // front-right-top
		glVertex3f(100.0,100.0,-100.0); // back-right-top
		glVertex3f(0.0,100.0,-100.0);   // back-left-top
	glEnd();
	glBegin(GL_LINES);
		glVertex3f(0.0,0.0,0.0);
		glVertex3f(0.0,100.0,0.0);
		glVertex3f(100.0,0.0,0.0);
		glVertex3f(100.0,100.0,0.0);
		glVertex3f(100.0,0.0,-100.0);
		glVertex3f(100.0,100.0,-100.0);
		glVertex3f(0.0,0.0,-100.0);
		glVertex3f(0.0,100.0,-100.0);
	glEnd();
}
