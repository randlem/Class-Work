/*
 * Guillaume Cottenceau (gc at mandrakesoft.com)
 *
 * Copyright 2002 MandrakeSoft
 *
 * This software may be freely redistributed under the terms of the GNU
 * public license.
 *
 */

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#define PNG_DEBUG 3
#include <png.h>

void abort_(const char * s, ...)
{
	va_list args;
	va_start(args, s);
	vfprintf(stderr, s, args);
	fprintf(stderr, "\n");
	va_end(args);
	abort();
}

int x, y;

int width=640, height=480;
png_byte color_type=PNG_COLOR_TYPE_RGBA;
png_byte bit_depth=32;

png_structp png_ptr;
png_infop info_ptr;
int number_of_passes;
png_bytep * row_pointers;

void write_png_file(char* file_name)
{
	/* create file */
	FILE *fp = fopen(file_name, "wb");
	if (!fp)
		abort_("[write_png_file] File %s could not be opened for writing", file_name);

	/* initialize stuff */
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr)
		abort_("[write_png_file] png_create_write_struct failed");

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
		abort_("[write_png_file] png_create_info_struct failed");

	//if (setjmp(png_jmpbuf(png_ptr)))
	//	abort_("[write_png_file] Error during init_io");

	png_init_io(png_ptr, fp);

	/* write header */
	//if (setjmp(png_jmpbuf(png_ptr)))
	//	abort_("[write_png_file] Error during writing header");

	png_set_IHDR(png_ptr, info_ptr, width, height,
		     bit_depth, color_type, PNG_INTERLACE_NONE,
		     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	/* write bytes */
	//if (setjmp(png_jmpbuf(png_ptr)))
	//	abort_("[write_png_file] Error during writing bytes");

	png_write_image(png_ptr, row_pointers);

	/* end write */
	//if (setjmp(png_jmpbuf(png_ptr)))
	//	abort_("[write_png_file] Error during end of write");

	png_write_end(png_ptr, NULL);

}

void process_file(void)
{
	if (info_ptr->color_type != PNG_COLOR_TYPE_RGBA)
		abort_("[process_file] color_type of input file must be PNG_COLOR_TYPE_RGBA (is %d)", info_ptr->color_type);

	for (y=0; y<height; y++) {
		png_byte* row = row_pointers[y];
		for (x=0; x<width; x++) {
			png_byte* ptr = &(row[x*4]);
            ptr[0] = (x*y)%255; ptr[1] = x%255; ptr[2] = y%255; ptr[3] = 0;
		}
	}

}

int main(int argc, char **argv)
{
    printf("asdf");

	//process_file();
	//write_png_file(argv[1]);
}
