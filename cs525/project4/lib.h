#ifndef __LIB_H__
#define __LIB_H__

typedef struct {
	float r; // red
	float g; // green
	float b; // blue
	float a; // alpha
} color;

// 16 color standard palette
const color WHITE		= { 1.00, 1.00, 1.00, 1.00 };
const color BLACK		= { 0.00, 0.00, 0.00, 1.00 };
const color DK_GRAY		= { 0.34, 0.34, 0.34, 1.00 };
const color LT_GRAY		= { 0.63, 0.63, 0.63, 1.00 };
const color DK_RED		= { 0.68, 0.14, 0.14, 1.00 };
const color RED			= { 1.00, 0.00, 0.00, 1.00 };
const color BLUE		= { 0.16, 0.29, 0.84, 1.00 };
const color LT_BLUE		= { 0.62, 0.69, 1.00, 1.00 };
const color CYAN		= { 0.16, 0.82, 0.82, 1.00 };
const color GREEN		= { 0.11, 0.41, 0.08, 1.00 };
const color LT_GREEN	= { 0.51, 0.77, 0.48, 1.00 };
const color BROWN		= { 0.51, 0.29, 0.10, 1.00 };
const color PURPLE		= { 0.51, 0.15, 0.75, 1.00 };
const color ORANGE		= { 1.00, 0.57, 0.20, 1.00 };
const color YELLOW		= { 1.00, 0.93, 0.20, 1.00 };
const color TAN			= { 0.91, 0.87, 0.73, 1.00 };
const color PINK		= { 1.00, 0.80, 0.95, 1.00 };

typedef struct {
	int x;
	int y;
} vertex_2d_t;

typedef struct {
	vertex_2d_t p1;
	vertex_2d_t p2;
} line_2d_t;

typedef struct {
	int bottom;
	int left;
	int top;
	int right;
} rect_t;

void draw_rect_filled(const rect_t &r, const color &c) {
	glColor4f(c.r, c.g, c.b, c.a);
	glPolygonMode(GL_FRONT,GL_FILL);
	glBegin(GL_POLYGON);
		glVertex2i(r.left,r.top);
		glVertex2i(r.left,r.bottom);
		glVertex2i(r.right,r.bottom);
		glVertex2i(r.right,r.top);
	glEnd();
}

void draw_rect_hollow(const rect_t &r, const color &c) {
	glColor4f(c.r, c.g, c.b, c.a);
	glBegin(GL_LINE_LOOP);
		glVertex2i(r.left,r.top);
		glVertex2i(r.left,r.bottom);
		glVertex2i(r.right,r.bottom);
		glVertex2i(r.right,r.top);
	glEnd();
}

/* Possible fonts:
	GLUT_BITMAP_9_BY_15,
	GLUT_BITMAP_8_BY_13,
	GLUT_BITMAP_TIMES_ROMAN_10,
	GLUT_BITMAP_TIMES_ROMAN_24,
	GLUT_BITMAP_HELVETICA_10,
	GLUT_BITMAP_HELVETICA_12,
	GLUT_BITMAP_HELVETICA_18
*/

// modify from code found at
//	http://www.opengl.org/resources/features/fontsurvey/sooft/examples/glutfonts.tar.gz
void print_string(int bottom, int left, char* s, void* font, const color &c)
{
	glColor4f(c.r, c.g, c.b, c.a);
	glRasterPos2f(left, bottom);
	if (s && strlen(s)) {
		while (*s) {
			glutBitmapCharacter(font, *s);
			s++;
		}
	}
}


#endif
