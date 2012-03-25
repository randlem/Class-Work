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
#include <mpi.h>

#define ImageHeight 1024
#define ImageWidth 1024

void write_png(char* filename);

int final_img[ImageHeight][ImageWidth];

int main(int argv, char* argc[]) {
    double xmin,xmax,ymin,ymax;
    double x_factor = (xmax-xmin)/(ImageWidth-1);
    double y_factor = (ymax-ymin)/(ImageHeight-1);
    unsigned int maxitr,y,x,n;
    double c_im,c_re,z_re,z_im,z_re2,z_im2;
    char inside = 0;
    int id, numb_proc;
    int part_size,part_root;
    int i,offset;
    int send_flg,recv_flg=1;
    int img[ImageHeight][ImageWidth];
    MPI_Status status;

    MPI_Init(&argv,&argc);

    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    MPI_Comm_size(MPI_COMM_WORLD,&numb_proc);

    if(argv < 7) {
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

    part_root = ImageHeight%(numb_proc-1);
    part_size = (ImageHeight-part_root)/(numb_proc-1);

    memset(final_img,0,sizeof(int)*ImageHeight*ImageWidth);
    memset(img,0,sizeof(int)*ImageHeight*ImageWidth);

    if(id == 0) {

        /* since it's possible for us to end up with extra lines i'm going to let
           the root node take care of them because we'll always have in the worst case
           partition_a-1 extra lines to calculate */
        if(part_root != 0) {
            for(y=ImageHeight-part_root; y<ImageHeight; y++) {
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
                    if(inside == 0) { final_img[y][x] = n; }
                    else { final_img[y][x] = 0; }
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        /* it's not the most elegant solution but it works...ordered send and recieve. Each child
           will block till the parent wants it to send the childs data. The parent will block till
           the child sends the data then proceede to the next child. */
        recv_flg = 1;
        for(i=1; i < numb_proc; i++) {
            MPI_Send(&recv_flg,1,MPI_INT,i,0,MPI_COMM_WORLD);
            MPI_Recv(img[(i-1)*part_size],part_size*ImageHeight,MPI_INT,
                        i,0,MPI_COMM_WORLD,&status);
        }

        memcpy(final_img,img,sizeof(int)*ImageHeight*ImageWidth);

    } else {

        /* process the data */
        offset=part_size*(id-1);
        for(y=offset; y < part_size+offset-1; y++) {
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

        /* wait for all the process to be done (yea i know this
           isn't needed, but it'll help with the timing) */
        MPI_Barrier(MPI_COMM_WORLD);

        /* block till the parent wants us to send...then send the data */
        MPI_Recv(&send_flg,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
        MPI_Send(img[offset],part_size*ImageWidth,MPI_INT,0,0,MPI_COMM_WORLD);

    }

    if(id == 0) {
        write_png(argc[6]);
    }

    MPI_Finalize();

    return(0);
}

void write_png(char* filename) {
    int x, y;
    int width=ImageWidth, height=ImageHeight;
    png_byte color_type=PNG_COLOR_TYPE_RGBA;
    png_byte bit_depth=8;
    png_structp png_ptr;
    png_infop info_ptr;
    int number_of_passes=1;
    png_bytep * row_pointers;
    FILE* fp;

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y<height; y++)
        row_pointers[y] = (png_byte*) malloc(width*((bit_depth/8)*4));

    for (y=0; y<height; y++) {
        png_byte* row = row_pointers[y];
        for (x=0; x<width; x++) {
            png_byte* ptr = &(row[x*4]);
            if(final_img[y][x] >= 1) {
                ptr[0] = 0; ptr[1] = final_img[y][x]%255; ptr[2] = 0; ptr[3] = 255;
            } else {
                ptr[0] = 0; ptr[1] = 0; ptr[2] = 0; ptr[3] = 255;
            }
        }
    }

    fp = fopen(filename, "wb");
    if(fp == NULL) {
        fprintf(stderr,"Couldn't open file.");
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
}
