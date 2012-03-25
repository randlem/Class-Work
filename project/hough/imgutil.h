#ifndef __IMGUTIL_H__
#define __IMGUTIL_H__

#include <string>
using std::string;

#include <stdio.h>
#include <png.h>

#include "util.h"
#include "gfxutil.h"

#define IMGUTIL_SIG_BYTES 8

typedef struct {
	uint_t		width;
	uint_t 		height;
	png_byte	color_type;
	png_byte	bit_depth;
	color_t**	pixels;
} img_t;

bool loadImage(string file_name, img_t* img) {
	uchar_t header[8];
	FILE *fp;
	png_structp png;
	png_infop info;
	png_infop end_info;
	png_bytep *row_pointers;
	int y, x;
	png_byte *row, *pxl;

	fp = fopen(file_name.c_str(),"rb");
	if (!fp)
		return handleError("Unable to find file!");

	fread(header, 1, IMGUTIL_SIG_BYTES, fp);

	if (png_sig_cmp(header, 0, IMGUTIL_SIG_BYTES))
		return handleError("File is not a valid PNG file!");

	png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
		NULL, NULL, NULL);
	if (!png)
		return false;

	info = png_create_info_struct(png);
	if (!info) {
		png_destroy_read_struct(&png, NULL, NULL);
		return false;
	}

	end_info = png_create_info_struct(png);
	if (!end_info) {
		png_destroy_read_struct(&png, &info, NULL);
		return false;
	}

	if (setjmp(png_jmpbuf(png))) {
		png_destroy_read_struct(&png, &info, NULL);
		return handleError("Error during init_io");
	}

	png_init_io(png, fp);
	png_set_sig_bytes(png, IMGUTIL_SIG_BYTES);
	png_read_info(png, info);

	img->width		= info->width;
	img->height		= info->height;
	img->color_type	= info->color_type;
	img->bit_depth	= info->bit_depth;

	if (img->bit_depth < 8)
        png_set_packing(png);
	if (img->bit_depth == 16)
        png_set_strip_16(png);

	if (img->color_type == PNG_COLOR_TYPE_GRAY ||
			img->color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
		png_set_gray_to_rgb(png);

	if (png_get_valid(png, info, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png);

	png_set_interlace_handling(png);
	png_read_update_info(png, info);

	img->width		= info->width;
	img->height		= info->height;
	img->color_type	= info->color_type;
	img->bit_depth	= info->bit_depth;

	debug("Loading PNG %s %dx%d (%dbbp)",file_name.c_str(),img->width,img->height,img->bit_depth);

	if (setjmp(png_jmpbuf(png))) {
		png_destroy_read_struct(&png, &info, NULL);
		return handleError("Error during read_img");
	}

	if (img->color_type != PNG_COLOR_TYPE_RGB &&
			img->color_type != PNG_COLOR_TYPE_RGB_ALPHA) {
		png_destroy_read_struct(&png, &info, NULL);
		return handleError("Files must be RGBA!");
	}

	row_pointers = new png_bytep[img->height];
	img->pixels  = new color_t*[img->height];
	for (y=0; y < img->height; y++) {
		row_pointers[y] = new png_byte[info->rowbytes];
		img->pixels[y] = new color_t[img->width];
	}

	png_read_image(png, row_pointers);

	for (y=0; y < img->height; y++) {
		row = row_pointers[y];
		for (x=0; x < img->width; x++) {
			pxl = &(row[x*3]);
			img->pixels[y][x].r = pxl[0];
			img->pixels[y][x].g = pxl[1];
			img->pixels[y][x].b = pxl[2];
			//img->pixels[y][x].a = pxl[3];
		}
	}

	fclose(fp);

	for (y=0; y < img->height; y++)
		delete [] row_pointers[y];
	delete [] row_pointers;

	return true;
}

bool writeImage(string file_name, img_t* img) {
	FILE *fp;
	png_structp png;
	png_infop info;
	png_infop end_info;
	png_bytep *row_pointers;
	int y, x;
	png_byte *row, *pxl;

	fp = fopen(file_name.c_str(), "wb");
	if (!fp)
		return handleError("Failed to open file for writing!");

	png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png)
		return handleError("Failed to create png out struct!");

	info = png_create_info_struct(png);
	if (!info)
		return handleError("Failed to create the png info struct!");

	if (setjmp(png_jmpbuf(png)))
		return handleError("Error during init_io!");

	png_init_io(png, fp);

	if (setjmp(png_jmpbuf(png)))
		return handleError("Error during png header writing!");

	png_set_IHDR(png, info, img->width, img->height, img->bit_depth,
		PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
		PNG_FILTER_TYPE_BASE);
	png_write_info(png,info);

	if (setjmp(png_jmpbuf(png)))
		return handleError("Error during png write!");

	row_pointers = new png_bytep[img->height];
	for (y=0; y < img->height; y++)
		row_pointers[y] = new png_byte[info->rowbytes];

	for (y=0; y < img->height; y++) {
		row = row_pointers[y];
		for (x=0; x < img->width; x++) {
			pxl = &(row[x*3]);
			pxl[0] = img->pixels[y][x].r;
			pxl[1] = img->pixels[y][x].g;
			pxl[2] = img->pixels[y][x].b;
			//pxl[3] = img->pixels[y][x].a;
		}
	}

	png_write_image(png, row_pointers);

	if (setjmp(png_jmpbuf(png)))
		return handleError("Error during png eof write!");

	png_write_end(png, NULL);

	for (y=0; y < img->height; y++)
		delete [] row_pointers[y];
	delete [] row_pointers;

	fclose(fp);

	return true;
}

#endif
