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
	int x_pos,y_pos;

	if(x < f->min_x || x > f->max_x)
		return;
	if(y < f->min_y || y > f->max_y)
		return;

	x_pos = (int)((x / (float)f->scale_x) + (f->width/2));
	y_pos = (int)((y / (float)f->scale_y) + (f->height/2));

	png_byte* row = f->row_pointers[y_pos];
	png_byte* ptr = &(row[x_pos*4]);
	ptr[0] = r; ptr[1] = g; ptr[2] = b; ptr[3] = 255;
}

void plot_line(png_file* f, double x0, double y0, double x1, double y1, uchar r, uchar g, uchar b) {
	int error;
	int dx,dy;
	int x0r,y0r,x1r,y1r;
	int x,y;
	int x_inc,y_inc;
	int i;

	x = x0r = (int)((x0 / (float)f->scale_x) + (f->width/2));
	y = y0r = (int)((y0 / (float)f->scale_y) + (f->height/2));
	x1r = (int)((x1 / (float)f->scale_x) + (f->width/2));
	y1r = (int)((y1 / (float)f->scale_y) + (f->height/2));

	dx = x1r - x0r;
	dy = y1r - y0r;

	x_inc = 1;
	if(dx < 0) {
		x_inc = -1;
		dx = -dx;
	}

	y_inc = 1;
	if(dy < 0) {
		y_inc = -1;
		dy = -dy;
	}

	if(dx > dy) {
		error = dy * 2 - dx;

		for(i=0; i < dx; i++) {
			png_byte* row = f->row_pointers[y];
			png_byte* ptr = &(row[x*4]);
			ptr[0] = r; ptr[1] = g; ptr[2] = b; ptr[3] = 255;

			if(error >= 0) {
				error -= dx * 2;
				y+=y_inc;
			}

			error += dy * 2;
			x+=x_inc;
		}

	} else {
		error = dx * 2 - dy;

		for(i=0; i < dy; i++) {
			png_byte* row = f->row_pointers[y];
			png_byte* ptr = &(row[x*4]);
			ptr[0] = r; ptr[1] = g; ptr[2] = b; ptr[3] = 255;

			if(error >= 0) {
				error -= dy * 2;
				x+=x_inc;
			}

			error += dx * 2;
			y+=y_inc;
		}
	}
}
