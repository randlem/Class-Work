#ifndef __UTIL_H__
#define __UTIL_H__

#ifndef DEBUG
#define DEBUG 0
#endif

#include <iostream>
using std::cerr;
using std::endl;

#include <string>
using std::string;

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#define ERROR_MESSAGE_SIZE	1024

void debug(const char * s, ...) {
	if (DEBUG > 0) {
		va_list args;
		va_start(args, s);
		vfprintf(stdout, s, args);
		fprintf(stdout, "\n");
		va_end(args);
	}
}

typedef unsigned char uchar_t;
typedef unsigned int  uint_t;

typedef struct {
	float r; // red
	float g; // green
	float b; // blue
	float a; // alpha
} color_t;

// 16 color standard palette
const color_t WHITE		= { 1.00, 1.00, 1.00, 1.00 };
const color_t BLACK		= { 0.00, 0.00, 0.00, 1.00 };
const color_t DK_GRAY	= { 0.34, 0.34, 0.34, 1.00 };
const color_t LT_GRAY	= { 0.63, 0.63, 0.63, 1.00 };
const color_t DK_RED	= { 0.68, 0.14, 0.14, 1.00 };
const color_t RED		= { 1.00, 0.00, 0.00, 1.00 };
const color_t BLUE		= { 0.16, 0.29, 0.84, 1.00 };
const color_t LT_BLUE	= { 0.62, 0.69, 1.00, 1.00 };
const color_t CYAN		= { 0.16, 0.82, 0.82, 1.00 };
const color_t GREEN		= { 0.11, 0.41, 0.08, 1.00 };
const color_t LT_GREEN	= { 0.51, 0.77, 0.48, 1.00 };
const color_t BROWN		= { 0.51, 0.29, 0.10, 1.00 };
const color_t PURPLE	= { 0.51, 0.15, 0.75, 1.00 };
const color_t ORANGE	= { 1.00, 0.57, 0.20, 1.00 };
const color_t YELLOW	= { 1.00, 0.93, 0.20, 1.00 };
const color_t TAN		= { 0.91, 0.87, 0.73, 1.00 };
const color_t PINK		= { 1.00, 0.80, 0.95, 1.00 };

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
void print_string(int bottom, int left, void* font, const color_t &c, const char* s, ...) {
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

char error_message[ERROR_MESSAGE_SIZE];

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

#endif
