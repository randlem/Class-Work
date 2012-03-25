#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <png.h>

#define X_RES 1024
#define Y_RES 1024

void write_png();
int mandint();

const int maxX=320,mx=320,maxY=200,my=199;
int maxit=1000,Xctr,Yctr;
double asp,dx,dy,xmin,ymin,xmax,ymax;
int img[320][200];

main() {
    int output,ret,flag=1;
    float scale=1.0;
    double cr,ci;

    Xctr=(maxX>>1); Yctr=(maxY>>1);
    mandint();
    write_png();

    return(0);
}

int mandint() {
    int col,j,k,colour,flag,old_colour,Xoffset,Yoffset;
    double cr,ci,jmax,x,y,xsq,ysq,distsq,xold,yold,delta;

    xmin=-2.0; xmax=0.8; ymin=-1.3; ymax=1.3;
    Xoffset=Xctr-(mx>>1);
    Yoffset=Yctr-(my>>1);
    dx=(xmax-xmin)/(mx-1);
    dy=(ymax-ymin)/(my-1);
    jmax=(my>>1);
    delta=((dx < dy ? dx : dy) / 2.0);
    for(j=0; j<=jmax; j++) {
        for(k=0; k<=mx; k++) {
            x=cr=xmin+k*dx;
            y=ci=ymin+j*dy;
            xsq=ysq=0.0;
            colour=0;
            xold=yold=0.0;

            while((colour < maxit) && (xsq+ysq < 4.0)) {
                xsq=x*x; ysq=y*y;
                y=2*x*y+ci;
                x=xsq-ysq+cr;
                if(old_colour == maxit)
                {
                    if((colour & 15) == 0)
                    {
                        xold=x;
                        yold=y;
                    } else
                        if((fabs(xold-x)+fabs(yold-y) < delta))
                            colour=maxit-1;
                }
                colour++;
            }
            if(colour < maxit)
                col=1;
            else
                col=0;
            img[Xoffset+k][Yoffset+j] = col;
            img[Xoffset+k][maxY-(Yoffset+j)] = col;

            old_colour=colour;
        }
    }
    return(0);
}

void write_png() {
    int x, y;
    int width=X_RES, height=Y_RES;
    png_byte color_type=PNG_COLOR_TYPE_RGBA;
    png_byte bit_depth=8;
    png_structp png_ptr;
    png_infop info_ptr;
    int number_of_passes=1;
    png_bytep * row_pointers;
    char filename[] = "output.png";

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y<height; y++)
        row_pointers[y] = (png_byte*) malloc(width*((bit_depth/8)*4));

    for (y=0; y<height; y++) {
        png_byte* row = row_pointers[y];
        for (x=0; x<width; x++) {
            png_byte* ptr = &(row[x*4]);
            if(img[y][x] == 1) {
                ptr[0] = 255; ptr[1] = 255; ptr[2] = 255; ptr[3] = 255;
            } else {
                ptr[0] = 0; ptr[1] = 0; ptr[2] = 0; ptr[3] = 255;
            }
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

}