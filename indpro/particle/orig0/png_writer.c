#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include "png_writer.h"

void open_file(png_file* f, char* filename, int width, int height, int min_x, int max_x, int min_y, int max_y) {
	int y, x;

	f->width = width;
	f->height = height;
	f->min_x = min_x;
	f->max_x = max_x;
	f->min_y = min_y;
	f->max_y = max_y;
	f->scale_x = (abs(f->min_x)+abs(f->max_x))/((float)f->width),
	f->scale_y = (abs(f->min_y)+abs(f->max_y))/((float)f->height);
    f->color_type = PNG_COLOR_TYPE_RGBA;
    f->bit_depth = 8;
    f->fp = fopen(filename, "wb");

    f->row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y < f->height; y++)
        f->row_pointers[y] = (png_byte*) malloc(f->width*((f->bit_depth/8) * 4));

    for (y = 0; y < f->height; y++) {
        png_byte* row = f->row_pointers[y];
        for (x = 0; x < f->width; x++) {
            png_byte* ptr = &(row[x*4]);
			ptr[0] = 0; ptr[1] = 0; ptr[2] = 0; ptr[3] = 255;
		}
    }
}

void write_file(png_file* f) {
	png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, f->fp);
    png_set_IHDR(png_ptr, info_ptr, f->width, f->height,
             f->bit_depth, f->color_type, PNG_INTERLACE_NONE,
             PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, f->row_pointers);
    png_write_end(png_ptr, NULL);
}

void close_file(png_file* f) {
	int y;

	fclose(f->fp);

    for (y=0; y < f->height; y++)
        free(f->row_pointers[y]);
	free(f->row_pointers);
}

void plot_point(png_file* f, double x, double y, uchar r, uchar g, uchar b) {
	float x_pos,y_pos;

	if(x < f->min_x || x > f->max_x)
		return;
	if(y < f->min_y || y > f->max_y)
		return;

	x_pos = (fabs(x) / (float)f->scale_x);
	y_pos = (fabs(y) / (float)f->scale_y);
	printf("%f %f\n",x_pos,y_pos);

	png_byte* row = f->row_pointers[(int)rint(y_pos)];
	png_byte* ptr = &(row[(int)rint(x_pos)*4]);
	ptr[0] = 255; ptr[1] = 255; ptr[2] = 255; ptr[3] = 255;
}
