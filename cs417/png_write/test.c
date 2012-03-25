#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <png.h>

int x, y;

int width=1020, height=1020;
png_byte color_type=PNG_COLOR_TYPE_RGBA;
png_byte bit_depth=8;

png_structp png_ptr;
png_infop info_ptr;
int number_of_passes=1;
png_bytep * row_pointers;

char filename[] = "test1.png";

int main() {
    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y<height; y++)
        row_pointers[y] = (png_byte*) malloc(width*((bit_depth/8)*4));

    for (y=0; y<height; y++) {
		png_byte* row = row_pointers[y];
		for (x=0; x<width; x++) {
			png_byte* ptr = &(row[x*4]);
                ptr[0] = (x*y)%255; ptr[1] = (x*y)%255; ptr[2] = (x*y)%255; ptr[3] = 255;
		}
	}

	FILE *fp = fopen(filename, "wb");
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	info_ptr = png_create_info_struct(png_ptr);
	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, width, height,
		     bit_depth, color_type, PNG_INTERLACE_NONE,
		     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	png_write_info(png_ptr, info_ptr);
	png_write_image(png_ptr, row_pointers);
	png_write_end(png_ptr, NULL);

    return(0);
}
