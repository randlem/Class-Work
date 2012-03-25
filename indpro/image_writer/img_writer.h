#include <png.h>

typedef unsigned char pixel_element;

typedef struct {
	pixel_element red;
	pixel_element green;
	pixel_element blue;
	pixel_element alpha;
} pixel;

typedef struct {
	unsigned int width;
	unsigned int height;
	pixel** image;
} image;

int create_image(int width, int height, image* img) {
	img->width = width;
	img->height = height;

	if(width%2 != 0)
		return(0);
	if(height%2 != 0)
		return(0);

