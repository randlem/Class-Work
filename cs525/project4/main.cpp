/*******************************************************************************
*
* Program #4 -- Line Clipping Example
*
* Mark Randles
* 2009-04-08
* CS525 - Computer Graphics
*
* DESC: This program is designed to draw part of an n-gon clipping region.  The
*	clipping region is defined by a bottom-left and top-right corner as selected
*	by the user using the mouse.  The lines inside the clipping region are drawn
*	using random colors.  The full rosette can be show at the same time.  The
*	clipping region is outlined in a blue box.  The center of the screen
*	(200,200) is marked by a green cross.
*
* ARCHITECTURE: The program is structured using the keyboard and mouse input to
*	drive various state machines.  One state machine gets the input from the
*	user to define the clipping region.  The other state machine drives the
*	actual drawing of the rosette in the clipped region.  The drawing of the
*	rosette is done using a set of lines.  Each line is clipped individually
*	then drawn.
*
* PROGRAM CONTROLS:
*	q/Q: Quit the program
*	<left>/<right>/<up>/<down>: Move clipping region by 10px
*	h/H: shrink/grow clipping region by 5px along the x-axis
*	j/J: shring/grow clipping region by 5px along the y-axis
*	f/F: toggle showing the full rosette while the rosette is being clipped
*	n/N: reset the program
*
* EXTRA FEATURES:
*	1) Allow user to shrink/grow the clipping region in 2d.
*	2) Allow user to reset the program.
*	3) Showing the user the full rosette after the clipping region is defined.
*	4) Text output on-screen telling the user their defined clipping region.
*
*******************************************************************************/
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

#include "cs_425_525_setup.h"
#include "lib.h"

#define WINDOW_X 400
#define WINDOW_Y 400
#define WINDOW_NAME "Program 4 -- Line Clipping Example"
#define PI 3.14159265

#define TOP		0x08
#define BOTTOM	0x04
#define LEFT	0x02
#define RIGHT	0x01

#define CLIP_REGION_MOVE 	10
#define CLIP_REGION_RESIZE	5

#define RANDRANGE(a,b) (rand() % ((b) - (a) + 1) + (a))
#define SWAP(a,b) ((a) ^= (b) ^= (a) ^= (b))

typedef unsigned char uchar_t;

vertex_2d_t ngon_vertex[7] = {
	{200, 350},
	{75, 300},
	{50, 175},
	{115, 50},
	{285, 50},
	{350, 175},
	{325, 300}
};

line_2d_t ngon_rosette[21] = {
	{ngon_vertex[0],ngon_vertex[1]},
	{ngon_vertex[0],ngon_vertex[2]},
	{ngon_vertex[0],ngon_vertex[3]},
	{ngon_vertex[0],ngon_vertex[4]},
	{ngon_vertex[0],ngon_vertex[5]},
	{ngon_vertex[0],ngon_vertex[6]},

	{ngon_vertex[1],ngon_vertex[2]},
	{ngon_vertex[1],ngon_vertex[3]},
	{ngon_vertex[1],ngon_vertex[4]},
	{ngon_vertex[1],ngon_vertex[5]},
	{ngon_vertex[1],ngon_vertex[6]},

	{ngon_vertex[2],ngon_vertex[3]},
	{ngon_vertex[2],ngon_vertex[4]},
	{ngon_vertex[2],ngon_vertex[5]},
	{ngon_vertex[2],ngon_vertex[6]},

	{ngon_vertex[3],ngon_vertex[4]},
	{ngon_vertex[3],ngon_vertex[5]},
	{ngon_vertex[3],ngon_vertex[6]},

	{ngon_vertex[4],ngon_vertex[5]},
	{ngon_vertex[4],ngon_vertex[6]},

	{ngon_vertex[5],ngon_vertex[6]},
};

void reset_clipper(void);
void display_func(void);
void keyboard_func(uchar_t, int, int);
void keyboard_special_func(int, int, int);
void mouse_func(int, int, int, int);
void idle_func(void);
void terminate(void);

void draw_crosshair(int, int, const color &);
void draw_ngon(const color &);
void draw_clipped_ngon(void);
bool clip_line(const rect_t &, line_2d_t *);
uchar_t generate_endcode(const rect_t &, vertex_2d_t &);

typedef struct {
	bool	set_bottom_left;
	bool	set_top_right;
	rect_t	rect;
} clipper_t;

static clipper_t	clipper;
const color line_colors[2] = { PURPLE, ORANGE };
static bool show_full_rosette = false;

int main(int argc, char* argv[]) {
	// setup the program
	reset_clipper();
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

void display_func(void) {
	char buffer[128];

	// black background
	glClearColor(BLACK.r, BLACK.g, BLACK.b, BLACK.a);
	glClear(GL_COLOR_BUFFER_BIT);

	// draw center crosshair
	draw_crosshair(200,200,GREEN);

	// run through the possible draw states
	if (!clipper.set_bottom_left && !clipper.set_top_right) {
		memset(buffer, 0, sizeof(char) * 128);
		sprintf(buffer, "Click to set the bottom-left corner");
		print_string(0, 0, buffer, GLUT_BITMAP_HELVETICA_18, WHITE);
		draw_ngon(DK_GRAY);
	} else if (clipper.set_bottom_left && !clipper.set_top_right) {
		memset(buffer, 0, sizeof(char) * 128);
		sprintf(buffer, "Click to set the top-right corner");
		print_string(0, 0, buffer, GLUT_BITMAP_HELVETICA_18, WHITE);
		draw_ngon(DK_GRAY);
		draw_crosshair(clipper.rect.left,clipper.rect.bottom,RED);
	} else {
		memset(buffer, 0, sizeof(char) * 128);
		sprintf(buffer, "Clipping rect = {t: %.3d, b: %.3d, l: %.3d, r: %.3d}",
			clipper.rect.top, clipper.rect.bottom,
			clipper.rect.left, clipper.rect.right);
		print_string(0, 0, buffer, GLUT_BITMAP_HELVETICA_18, WHITE);
		draw_clipped_ngon();
		draw_rect_hollow(clipper.rect,BLUE);
	}

	// flush & swap the buffer
	glFlush();
	glutSwapBuffers();
}

void idle_func(void) {

}

void keyboard_func (uchar_t c, int x, int y) {
	// switch on the keyboard character pressed
	switch (c) {
		case 'N':
		case 'n': {
			reset_clipper();
		} break;
		case 'H':{
			clipper.rect.left -= CLIP_REGION_RESIZE;
		} break;
		case 'h': {
			if ((clipper.rect.right - clipper.rect.left - CLIP_REGION_RESIZE)
					> CLIP_REGION_RESIZE)
				clipper.rect.left += CLIP_REGION_RESIZE;
		} break;
		case 'J': {
			clipper.rect.top += CLIP_REGION_RESIZE;
		} break;
		case 'j': {
			if ((clipper.rect.top - clipper.rect.bottom - CLIP_REGION_RESIZE)
					> CLIP_REGION_RESIZE)
				clipper.rect.top -= CLIP_REGION_RESIZE;
		} break;
		case 'F':
		case 'f': {
			show_full_rosette = (show_full_rosette) ? false : true;
		} break;
		case 'Q': // quit the program
		case 'q': {
			terminate();
			exit(0);
		} break;
	}

	glutPostRedisplay();
}

void keyboard_special_func (int c, int x, int y) {
	// switch on the keyboard special key pressed
	switch (c) {
		case GLUT_KEY_UP: {
			clipper.rect.top	+= CLIP_REGION_MOVE;
			clipper.rect.bottom	+= CLIP_REGION_MOVE;
		} break;
		case GLUT_KEY_DOWN: {
			clipper.rect.top	-= CLIP_REGION_MOVE;
			clipper.rect.bottom	-= CLIP_REGION_MOVE;
		} break;
		case GLUT_KEY_LEFT: {
			clipper.rect.left	-= CLIP_REGION_MOVE;
			clipper.rect.right	-= CLIP_REGION_MOVE;
		} break;
		case GLUT_KEY_RIGHT: {
			clipper.rect.left	+= CLIP_REGION_MOVE;
			clipper.rect.right	+= CLIP_REGION_MOVE;
		} break;
	}

	glutPostRedisplay();
}

void mouse_func (int button, int state, int x, int y) {
	int t;

	switch (button) {
		case GLUT_LEFT_BUTTON: {
			switch (state) {
				case GLUT_DOWN: {
					if (!clipper.set_bottom_left) {
						clipper.rect.left		= x;
						clipper.rect.bottom		= WINDOW_Y - y;
					} else if (!clipper.set_top_right) {
						clipper.rect.right		= x;
						clipper.rect.top		= WINDOW_Y - y;

						if (clipper.rect.left > clipper.rect.right) {
							t 					= clipper.rect.left;
							clipper.rect.left	= clipper.rect.right;
							clipper.rect.right	= t;
						} else if (clipper.rect.bottom > clipper.rect.top) {
							t 					= clipper.rect.top;
							clipper.rect.top	= clipper.rect.bottom;
							clipper.rect.bottom	= t;
						}
					}
				} break;
				case GLUT_UP: {
					if (!clipper.set_bottom_left) {
						clipper.set_bottom_left = true;
						glutPostRedisplay();
					} else if (!clipper.set_top_right) {
						clipper.set_top_right	= true;
						glutPostRedisplay();
					}
				} break;
			}
		} break;
		case GLUT_MIDDLE_BUTTON:
		case GLUT_RIGHT_BUTTON:
		default: {

		}
	}
}

void terminate(void) {

}

void reset_clipper(void) {
	memset(&clipper.rect,0,sizeof(rect_t));
	clipper.set_bottom_left = false;
	clipper.set_top_right = false;
}

void draw_crosshair(int x, int y, const color &c) {
	glColor4f(c.r,c.g,c.b,c.a);
	glBegin(GL_POINTS);
	for(int dx=x-3; dx <= x+3; dx++)
		for(int dy=y-3; dy <= y+3; dy++)
			if (dy == y || dx == x)
				glVertex2i(dx,dy);
	glEnd();
}

void draw_ngon(const color &c) {
	glColor4f(c.r,c.g,c.b,c.a);
	for(int i=0; i < 21; i++) {
		glBegin(GL_LINES);
			glVertex2i(ngon_rosette[i].p1.x,ngon_rosette[i].p1.y);
			glVertex2i(ngon_rosette[i].p2.x,ngon_rosette[i].p2.y);
		glEnd();
	}
}

void draw_clipped_ngon() {
	line_2d_t *ngon_rosette_clipped = NULL;
	int rand_color;

	ngon_rosette_clipped = new line_2d_t[21];
	memcpy(ngon_rosette_clipped,ngon_rosette,sizeof(line_2d_t)*21);

	if (show_full_rosette)
		draw_ngon(DK_GRAY);

	for(int i=0; i < 21; i++) {
		if (clip_line(clipper.rect,&ngon_rosette_clipped[i])) {
			rand_color = RANDRANGE(0,1);
			glColor4f(
				line_colors[rand_color].r,
				line_colors[rand_color].g,
				line_colors[rand_color].b,
				line_colors[rand_color].a
			);
			glBegin(GL_LINES);
				glVertex2i(ngon_rosette_clipped[i].p1.x,
					ngon_rosette_clipped[i].p1.y);
				glVertex2i(ngon_rosette_clipped[i].p2.x,
					ngon_rosette_clipped[i].p2.y);
			glEnd();
		}
	}

	delete [] ngon_rosette_clipped;
}

bool clip_line(const rect_t &region, line_2d_t *l) {
	uchar_t endcode_1, endcode_2, endcode;
	int x,y;

	do {
		endcode_1 = generate_endcode(region, l->p1);
		endcode_2 = generate_endcode(region, l->p2);

		if ((endcode_1 & endcode_2) != 0) // trivial reject
			return false;
		if ((endcode_1 | endcode_2) == 0) // trivial accept
			return true;

		if (endcode_1 == 0)
			endcode = endcode_2;
		else
			endcode = endcode_1;

		if ((TOP & endcode) != 0) {
			x = l->p1.x + (l->p2.x - l->p1.x) * (region.top - l->p1.y) / (l->p2.y - l->p1.y);
			y = region.top;
		} else if ((BOTTOM & endcode) != 0) {
			x = l->p1.x + (l->p2.x - l->p1.x) * (region.bottom - l->p1.y) / (l->p2.y - l->p1.y);
			y = region.bottom;
		}

		if ((LEFT & endcode) != 0) {
			y = l->p1.y + (l->p2.y - l->p1.y) * (region.left - l->p1.x) / (l->p2.x - l->p1.x);
			x = region.left;
		} else if ((RIGHT & endcode) != 0) {
			y = l->p1.y + (l->p2.y - l->p1.y) * (region.right - l->p1.x) / (l->p2.x - l->p1.x);
			x = region.right;
		}

		if (endcode_1 == endcode) {
			l->p1.x = x;
			l->p1.y = y;
		} else {
			l->p2.x = x;
			l->p2.y = y;
		}

	} while (true);
}

uchar_t generate_endcode(const rect_t &region, vertex_2d_t &p) {
	uchar_t ret = 0;

	if (p.y > region.top)
		ret = TOP;
	else if (p.y < region.bottom)
		ret = BOTTOM;

	if (p.x < region.left)
		ret |= LEFT;
	else if (p.x > region.right)
		ret |= RIGHT;

	return ret;
}
