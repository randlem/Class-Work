#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <png.h>
#include "vector.h"

#define IMG_HEIGHT 1024
#define IMG_WIDTH 1024
#define MIN_X_IMG -200
#define MIN_Y_IMG -200
#define MAX_X_IMG  200
#define MAX_Y_IMG  200

void write_file_z(char* filename);
void write_file_y(char* filename);
void write_file_fake(char* filename);

vector *particles;
int num_particles;

int main(int argc, char* argv[]) {
	int i,file_cnt=0;
	char buffer[80];

	while(scanf("%i",&num_particles) != EOF) {
		particles = (vector*)malloc(sizeof(vector)*num_particles);

		for(i=0; i < num_particles; i++) {
			scanf("%f %f %f",&(particles[i].elements[0]),&(particles[i].elements[1]),&(particles[i].elements[2]));
		}

		sprintf(buffer,"%03i.png",file_cnt);
		write_file_y(buffer);
		file_cnt++;

		free(particles);
	}

	return(0);
}

void write_file_y(char* filename) {
    int x, y, x_pos, y_pos,i;
    int width=IMG_WIDTH, height=IMG_HEIGHT;
	float scale_x=(abs(MIN_X_IMG)+abs(MAX_X_IMG))/((float)IMG_WIDTH),
		  scale_y=(abs(MIN_Y_IMG)+abs(MAX_Y_IMG))/((float)IMG_HEIGHT);
    png_byte color_type=PNG_COLOR_TYPE_RGBA;
    png_byte bit_depth=8;
    png_structp png_ptr;
    png_infop info_ptr;
    png_bytep * row_pointers;

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y<height; y++)
        row_pointers[y] = (png_byte*) malloc(width*((bit_depth/8)*4));

	/* zero out the image buffer */
    for (y=0; y<height; y++) {
        png_byte* row = row_pointers[y];
        for (x=0; x<width; x++) {
            png_byte* ptr = &(row[x*4]);
			ptr[0] = 0; ptr[1] = 0; ptr[2] = 0; ptr[3] = 255;
		}
    }

	/* draw the center */
	x_pos = (int)(((float)abs(MIN_X_IMG)) / (float)scale_x);
	y_pos = (int)(((float)abs(MIN_Y_IMG)) / (float)scale_y);

	if(x_pos > 0 && x_pos < IMG_WIDTH-1 && y_pos > 0 && y_pos < IMG_HEIGHT-1) {
		png_byte* row = row_pointers[y_pos];
		png_byte* ptr = &(row[x_pos*4]);
		ptr[0] = 255; ptr[1] = 255; ptr[2] = 255; ptr[3] = 255;
	}

	for(i=0; i < num_particles; i++) {
		if((particles[i].elements[0] > MIN_X_IMG && particles[i].elements[0] < MAX_X_IMG) &&
		   (particles[i].elements[1] > MIN_Y_IMG && particles[i].elements[1] < MAX_Y_IMG)) {
			x_pos = (int)(((float)abs(MIN_X_IMG) + particles[i].elements[0]) / (float)scale_x);
			y_pos = ((int)(((float)abs(MIN_Y_IMG) + particles[i].elements[1]) / (float)scale_y));

			if(x_pos > 0 && x_pos < IMG_WIDTH-1 && y_pos > 0 && y_pos < IMG_HEIGHT-1) {
				png_byte* row = row_pointers[y_pos];
				png_byte* ptr = &(row[x_pos*4]);
				ptr[0] = 255; ptr[1] = 0; ptr[2] = 0; ptr[3] = 255;
			}
		}
	}

	/* actually write the file */
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
	fclose(fp);

	/* clean up my memory */
    for (y=0; y< height; y++)
        free(row_pointers[y]);
	free(row_pointers);
}

void write_file_z(char* filename) {
    int x, y, x_pos, y_pos,i;
    int width=IMG_WIDTH, height=IMG_HEIGHT;
	float scale_x=(abs(MIN_X_IMG)+abs(MAX_X_IMG))/((float)IMG_WIDTH),
		  scale_y=(abs(MIN_Y_IMG)+abs(MAX_Y_IMG))/((float)IMG_HEIGHT);
    png_byte color_type=PNG_COLOR_TYPE_RGBA;
    png_byte bit_depth=8;
    png_structp png_ptr;
    png_infop info_ptr;
    png_bytep * row_pointers;

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y<height; y++)
        row_pointers[y] = (png_byte*) malloc(width*((bit_depth/8)*4));

	/* zero out the image buffer */
    for (y=0; y<height; y++) {
        png_byte* row = row_pointers[y];
        for (x=0; x<width; x++) {
            png_byte* ptr = &(row[x*4]);
			ptr[0] = 0; ptr[1] = 0; ptr[2] = 0; ptr[3] = 255;
		}
    }

	/* draw the center */
	x_pos = (int)(((float)abs(MIN_X_IMG)) / (float)scale_x);
	y_pos = (int)(((float)abs(MIN_Y_IMG)) / (float)scale_y);

	if(x_pos > 0 && x_pos < IMG_WIDTH-1 && y_pos > 0 && y_pos < IMG_HEIGHT-1) {
		png_byte* row = row_pointers[y_pos];
		png_byte* ptr = &(row[x_pos*4]);
		ptr[0] = 255; ptr[1] = 255; ptr[2] = 255; ptr[3] = 255;
	}

	for(i=0; i < num_particles; i++) {
		if((particles[i].elements[0] > MIN_X_IMG && particles[i].elements[0] < MAX_X_IMG) &&
		   (particles[i].elements[2] > MIN_Y_IMG && particles[i].elements[2] < MAX_Y_IMG)) {
			x_pos = (int)(((float)abs(MIN_X_IMG) + particles[i].elements[0]) / (float)scale_x);
			y_pos = ((int)(((float)abs(MIN_Y_IMG) + particles[i].elements[2]) / (float)scale_y));

			if(x_pos > 0 && x_pos < IMG_WIDTH-1 && y_pos > 0 && y_pos < IMG_HEIGHT-1) {
				png_byte* row = row_pointers[y_pos];
				png_byte* ptr = &(row[x_pos*4]);
				ptr[0] = 255; ptr[1] = 0; ptr[2] = 0; ptr[3] = 255;
			}
		}
	}

	/* actually write the file */
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
	fclose(fp);

	/* clean up my memory */
	for (y=0; y<height; y++)
        free(row_pointers[y]);
	free(row_pointers);
}

void write_file_fake(char* filename) {
	int i;

	for(i=0; i < num_particles; i++) {
		printf("%f %f %f\n",particles[i].elements[0],particles[i].elements[1],particles[i].elements[2]);
	}
	printf("\n");
}
