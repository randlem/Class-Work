/****************************************
* mandle_single.c -- Mandelbrot Set Generator
*
* Mark Randles
* Parts adapted from :
*    http://www.students.tut.fi/~warp/Mandelbrot/
****************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <png.h>

#define ImageHeight 1024
#define ImageWidth 1024

void write_png();

int img[ImageHeight][ImageWidth];

int main(int argv, char* argc[]) {
    double xmin,xmax,ymin,ymax;
    double x_factor = (xmax-xmin)/(ImageWidth-1);
    double y_factor = (ymax-ymin)/(ImageHeight-1);
    unsigned int maxitr,y,x,n;
    double c_im,c_re,z_re,z_im,z_re2,z_im2;
    char inside = 0;

    if(argv < 6) {
        fprintf(stderr,"Too few parameters passed.\n");
        exit(1);
    }

    xmin = atof(argc[1]);
    xmax = atof(argc[2]);
    ymin = atof(argc[3]);
    ymax = atof(argc[4]);
    maxitr = atoi(argc[5]);

    x_factor = (xmax-xmin)/(ImageWidth-1);
    y_factor = (ymax-ymin)/(ImageHeight-1);

    for(y=0; y<ImageHeight; y++) {
        c_im = ymax - y*y_factor;
        for(x=0; x<ImageWidth; x++){
            c_re = xmin + x*x_factor;
            z_re = c_re;
            z_im = c_im;
            inside = 1;
            for(n=0; n<maxitr; n++) {
                z_re2 = z_re*z_re;
                z_im2 = z_im*z_im;
                if(z_re2 + z_im2 > 4) {
                    inside = 0;
                    break;
                }
                z_im = 2*z_re*z_im + c_im;
                z_re = z_re2 - z_im2 + c_re;
            }
            if(inside == 0) { img[y][x] = n; }
            else { img[y][x] = 0; }
        }
    }

    write_png();

    return(0);
}

void write_png() {
    int x, y;
    int width=ImageWidth, height=ImageHeight;
    png_byte color_type=PNG_COLOR_TYPE_RGBA;
    png_byte bit_depth=8;
    png_structp png_ptr;
    png_infop info_ptr;
    int number_of_passes=1;
    png_bytep * row_pointers;
    char filename[] = "output.png";
    FILE* fp;

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y<height; y++)
        row_pointers[y] = (png_byte*) malloc(width*((bit_depth/8)*4));

    for (y=0; y<height; y++) {
        png_byte* row = row_pointers[y];
        for (x=0; x<width; x++) {
            png_byte* ptr = &(row[x*4]);
            if(img[y][x] >= 1) {
                ptr[0] = 0; ptr[1] = img[y][x]%255; ptr[2] = 0; ptr[3] = 255;
            } else {
                ptr[0] = 0; ptr[1] = 0; ptr[2] = 0; ptr[3] = 255;
            }
        }
    }

    fp = fopen(filename, "wb");
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
