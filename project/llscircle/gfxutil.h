#ifndef __GFXUTIL_H__
#define __GFXUTIL_H__

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include "util.h"

typedef union {
	struct {
		uchar_t r;
		uchar_t g;
		uchar_t b;
		uchar_t a;
	};
	uint_t  rgba;
} color_t;

typedef struct {
	float r; // red
	float g; // green
	float b; // blue
	float a; // alpha
} color_gl_t;

// 16 color standard palette in GL colors
const color_gl_t GLC_WHITE		= { 1.00, 1.00, 1.00, 1.00 };
const color_gl_t GLC_BLACK		= { 0.00, 0.00, 0.00, 1.00 };
const color_gl_t GLC_DK_GRAY	= { 0.34, 0.34, 0.34, 1.00 };
const color_gl_t GLC_LT_GRAY	= { 0.63, 0.63, 0.63, 1.00 };
const color_gl_t GLC_DK_RED		= { 0.68, 0.14, 0.14, 1.00 };
const color_gl_t GLC_RED		= { 1.00, 0.00, 0.00, 1.00 };
const color_gl_t GLC_BLUE		= { 0.16, 0.29, 0.84, 1.00 };
const color_gl_t GLC_LT_BLUE	= { 0.62, 0.69, 1.00, 1.00 };
const color_gl_t GLC_CYAN		= { 0.16, 0.82, 0.82, 1.00 };
const color_gl_t GLC_GREEN		= { 0.11, 0.41, 0.08, 1.00 };
const color_gl_t GLC_LT_GREEN	= { 0.51, 0.77, 0.48, 1.00 };
const color_gl_t GLC_BROWN		= { 0.51, 0.29, 0.10, 1.00 };
const color_gl_t GLC_PURPLE		= { 0.51, 0.15, 0.75, 1.00 };
const color_gl_t GLC_ORANGE		= { 1.00, 0.57, 0.20, 1.00 };
const color_gl_t GLC_YELLOW		= { 1.00, 0.93, 0.20, 1.00 };
const color_gl_t GLC_TAN		= { 0.91, 0.87, 0.73, 1.00 };
const color_gl_t GLC_PINK		= { 1.00, 0.80, 0.95, 1.00 };

const color_t WHITE				= { 255, 255, 255, 255 };
const color_t BLACK				= {   0,   0,   0, 255 };
const color_t DK_GRAY			= {  87,  87,  87, 255 };
const color_t LT_GRAY			= { 161, 161, 161, 255 };
const color_t DK_RED			= { 173,  36,  36, 255 };
const color_t RED				= { 255,   0,   0, 255 };
const color_t BLUE				= {  41,  74, 214, 255 };
const color_t LT_BLUE			= { 158, 176, 255, 255 };
const color_t CYAN				= {  41, 209, 209, 255 };
const color_t GREEN				= {  28, 105,  20, 255 };
const color_t LT_GREEN			= { 130, 196, 122, 255 };
const color_t BROWN				= { 130,  74,  26, 255 };
const color_t PURPLE			= { 130,  38, 191, 255 };
const color_t ORANGE			= { 255, 145,  51, 255 };
const color_t YELLOW			= { 255, 237,  51, 255 };
const color_t TAN				= { 232, 222, 186, 255 };
const color_t PINK				= { 255, 204, 242, 255 };

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
void print_string(int bottom, int left, void* font, const color_gl_t &c, const char* s, ...) {
	va_list args;
	int buffer_len = strlen(s) * 2;
	char* buffer;

	buffer = new char[buffer_len];
	memset(buffer,'\0',sizeof(char)*buffer_len);

	va_start(args, s);
	vsprintf(buffer, s, args);
	va_end(args);

	glColor4f(c.r, c.g, c.b, c.a);
	glRasterPos2f(left, bottom);
	if (buffer && strlen(buffer)) {
		while (*buffer) {
			glutBitmapCharacter(font, *buffer);
			buffer++;
		}
	}
}

#endif
