#include "Image.h"

Image::Image(int w, int h, string fN) : raster(NULL), width(w), height(h), fileName(fN) {

	if(width < 0)
		width = 0;

	if(height < 0)
		height = 0;

	if(fileName.size() <= 0)
		fileName = "a";

	if(width > 0 && height > 0) {
		raster = new rgb*[height];
		for(int i=0; i < height; ++i)
			raster[i] = new rgb[width];
	}
}

Image::~Image() {
	if(raster != NULL) {
		for(int i=0; i < height; ++i)
			delete [] raster[i];
		delete [] raster;
	}
}

void Image::clearImage(rgb& color) {
	point p0 = { 0, 0 },
	      p1 = { width-1, height-1 };
	drawRect(p0,p1,color);
}

void Image::setFileName(string s) {
	fileName = s;
}

void Image::drawPoint(point& p, rgb& color) {
	if(p.x > width || p.x < 0 || p.y > height || p.y < 0)
		return;
	raster[p.y][p.x] = color;
}

void Image::drawLine(point& start, point& end, rgb& color) {
	int dx,dy;
	float m, t = 0.5;

	dx = end.x - start.x;
	dy = end.y - start.y;

	raster[start.y][start.x] = color;
	if(abs(dx) > abs(dy)) {
		m = (float) dy / (float) dx;
		t += start.y;
		dx = (dx < 0) ? -1 : 1;
		m *= dx;
		while(start.x != end.x) {
			start.x += dx;
			t += m;
			raster[(int)t][start.x] = color;
		}
	} else {
		m = (float) dx / (float) dy;
		t += start.x;
		dy = (dy < 0) ? -1 : 1;
		m *= dy;
		while(start.y != end.y) {
			start.y += dy;
			t += m;
			raster[start.y][(int)t] = color;
		}
	}
}

void Image::drawLine(point& start, point& end, int width, rgb& color) {
	int dx,dy;
	float m, t = 0.5;

	dx = end.x - start.x;
	dy = end.y - start.y;

	raster[start.y][start.x] = color;
	if(abs(dx) > abs(dy)) {
		m = (float) dy / (float) dx;
		t += start.y;
		dx = (dx < 0) ? -1 : 1;
		m *= dx;
		while(start.x != end.x) {
			start.x += dx;
			t += m;
			raster[(int)t][start.x] = color;
		}
	} else {
		m = (float) dx / (float) dy;
		t += start.x;
		dy = (dy < 0) ? -1 : 1;
		m *= dy;
		while(start.y != end.y) {
			start.y += dy;
			t += m;
			raster[start.y][(int)t] = color;
		}
	}
}

void Image::drawRect(point& topLeft, point& bottomRight, rgb& color) {
	int x,y;

	if(bottomRight.x < topLeft.x) {
		int tmp = topLeft.x;
		topLeft.x = bottomRight.x; bottomRight.x = tmp;
	}
	if(bottomRight.y < topLeft.y) {
		int tmp = topLeft.y;
		topLeft.y = bottomRight.y; bottomRight.y = tmp;
	}

	for(y=topLeft.y; y <= bottomRight.y; ++y)
		for(x=topLeft.x; x <= bottomRight.x; ++x)
			raster[y][x] = color;
}

bool Image::outputImage(ImageType outputType) {
	switch(outputType) {
		case PNG:
			return(outputPNG(fileName + ".png"));
	}
	return(false);
}

bool Image::outputPNG(string fileName) {
	FILE* fp;
	int x, y;
    png_byte color_type=PNG_COLOR_TYPE_RGBA;
    png_byte bit_depth=8;
    png_structp png_ptr;
    png_infop info_ptr;
    int number_of_passes=1;
    png_bytep * row_pointers;

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y < height; y++)
        row_pointers[y] = (png_byte*) malloc(width*((bit_depth/8)*4));

    for (y=0; y < height; y++) {
        png_byte* row = row_pointers[y];
        for (x=0; x<width; x++) {
            png_byte* ptr = &(row[x*4]);
            ptr[0] = raster[y][x].r; ptr[1] = raster[y][x].g; ptr[2] = raster[y][x].b; ptr[3] = 255;
        }
    }

	fp = fopen(fileName.c_str(), "wb");
    if(fp == NULL) {
		return(false);
    } else {
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        info_ptr = png_create_info_struct(png_ptr);
        png_init_io(png_ptr, fp);
        png_set_IHDR(png_ptr, info_ptr, width, height,
                 bit_depth, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
        png_write_info(png_ptr, info_ptr);
        png_write_image(png_ptr, row_pointers);
        png_write_end(png_ptr, NULL);
    }
	fclose(fp);

	return(true);
}
