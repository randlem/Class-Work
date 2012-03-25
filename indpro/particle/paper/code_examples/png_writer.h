#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>

#ifndef __PNG_WRITER_H__
#define __PNG_WRITER_H__

typedef unsigned char uchar;

typedef struct
{
	FILE* fp;
    int width;
	int height;
	int min_x;
	int max_x;
	int min_y;
	int max_y;
	float scale_x;
	float scale_y;
    png_byte color_type;
    png_byte bit_depth;
    png_bytep* row_pointers;
} png_file;

void open_file(png_file* f, char* filename, int width, int height, int min_x, int max_x, int min_y, int max_y);
void write_file(png_file* f);
void close_file(png_file* f);

void plot_point(png_file* f, double x, double y, uchar r, uchar g, uchar b);
void plot_line(png_file* f, double x0, double y0, double x1, double y1, uchar r, uchar g, uchar b);

#endif
