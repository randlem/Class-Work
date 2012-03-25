/******************************************************************************
* Particle Explosing Sim
*
* Mark Randles
* CS417
* Spring 2004
* Dr. Hassan Rajaei
*
* PROBLEM DESC: To simulate the explosion of a aggrate in a vaccum.  Each piece
* of the aggrate will be simulated with a point mass, each with a different
* mass.  The explosing will be from a singularity, each point mass emination
* from the same point, but with a outward velocity vector.  There will be a
* gravatational interaction with at least one body.  There will be no "ground".
* Each pieces motion will be tracked in 3 dimensions.
*
* GOAL: To test the speedup of the problem on a MPI/OMP hybred system vs. a MPI
* implementation, a OpenMP system, and a single process system.
*
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <png.h>
#include "vector.h"
#include "particle.h"

#define G 0.0000000000667 /* N*m^2/kg^2 */
#define e 0.0000000000001

//#define NUM_PARTICLES 20000
#define TIME 5
#define CYCLE_LIMIT 25
#define NUM_PARTICLES 1000
//#define TIME_LIMIT 10

#define IMG_HEIGHT 1024
#define IMG_WIDTH 1024
#define MIN_X_IMG -1000
#define MIN_Y_IMG -1000
#define MAX_X_IMG  1000
#define MAX_Y_IMG  1000

#define RAND_RANGE(a,b) ((a)+(rand()%((b)-(a)+1)))

particle center_of_mass;
particle* particles_old,*particles_new;

void write_file();

int main(int argc, char* argv[]) {
	int i,j;
	float t=0;
	int file_cnt;
	char filename[80];

	srand((int)time(NULL));

	particles_old = (particle*)malloc(sizeof(particle)*NUM_PARTICLES);
	particles_new = (particle*)malloc(sizeof(particle)*NUM_PARTICLES);

	center_of_mass.id = 0;
	center_of_mass.mass = 9999999999;
	vector_init(&center_of_mass.velocity);
	vector_init(&center_of_mass.acceleration);
	vector_init(&center_of_mass.position);

	for(i=0; i < NUM_PARTICLES; i++) {
		particles_old[i].id = i+1;

		particles_old[i].velocity.elements[0] = RAND_RANGE(0,0);
		particles_old[i].velocity.elements[1] = RAND_RANGE(0,0);
		particles_old[i].velocity.elements[2] = RAND_RANGE(0,0);

		particles_old[i].position.elements[0] = RAND_RANGE(-1000,1000);
		particles_old[i].position.elements[1] = RAND_RANGE(-1000,1000);
		particles_old[i].position.elements[2] = RAND_RANGE(-1000,1000);

		vector_init(&particles_old[i].acceleration);

		particles_old[i].mass = RAND_RANGE(1, 100);
	}

	file_cnt = 0;
	for(t=0; t < CYCLE_LIMIT; t++){
		memcpy(particles_new,particles_old,sizeof(particle)*NUM_PARTICLES);

		for(i=0; i < NUM_PARTICLES; i++) {
			double delta_x,delta_y,delta_z,
				  g;
			vector Fg;

			vector_init(&Fg);

			g = (G * particles_old[i].mass * center_of_mass.mass);

			delta_x = particles_old[i].position.elements[0]-center_of_mass.position.elements[0];
			delta_y = particles_old[i].position.elements[1]-center_of_mass.position.elements[1];
			delta_z = particles_old[i].position.elements[2]-center_of_mass.position.elements[2];

			if(delta_x != 0)
				Fg.elements[0] = (particles_old[i].position.elements[0] < center_of_mass.position.elements[0]) ? (float)((float)g/(float)(delta_x * delta_x)) : (-1.0)*(float)((float)g/(float)(delta_x * delta_x)); /* Fg in x-y plane */
			if(delta_y != 0)
				Fg.elements[1] = (particles_old[i].position.elements[1] < center_of_mass.position.elements[1]) ? (float)((float)g/(float)(delta_y * delta_y)) : (-1.0)*(float)((float)g/(float)(delta_y * delta_y)); /* Fg in x-z plane */
			if(delta_z != 0)
				Fg.elements[2] = (particles_old[i].position.elements[2] < center_of_mass.position.elements[2]) ? (float)((float)g/(float)(delta_z * delta_z)) : (-1.0)*(float)((float)g/(float)(delta_z * delta_z)); /* Fg in y-z plane */

			for(j=0; j < NUM_PARTICLES; j++) {
				if(j != i) {
					g = (G * particles_old[i].mass * particles_old[j].mass);

					delta_x = particles_old[i].position.elements[0] - particles_old[j].position.elements[0];
					delta_y = particles_old[i].position.elements[1] - particles_old[j].position.elements[1];
					delta_z = particles_old[i].position.elements[2] - particles_old[j].position.elements[2];

				if(delta_x != 0)
					Fg.elements[0] += (particles_old[i].position.elements[0] < particles_old[j].position.elements[0]) ? (float)((float)g/(float)(delta_x * delta_x)) : (-1.0)*(float)((float)g/(float)(delta_x * delta_x)); /* Fg in x-y plane */
				if(delta_y != 0)
					Fg.elements[1] += (particles_old[i].position.elements[1] < particles_old[j].position.elements[1]) ? (float)((float)g/(float)(delta_y * delta_y)) : (-1.0)*(float)((float)g/(float)(delta_y * delta_y)); /* Fg in x-z plane */
				if(delta_z != 0)
					Fg.elements[2] += (particles_old[i].position.elements[2] < particles_old[j].position.elements[2]) ? (float)((float)g/(float)(delta_z * delta_z)) : (-1.0)*(float)((float)g/(float)(delta_z * delta_z)); /* Fg in y-z plane */
				}
			}

			update_acceleration(&particles_new[i],&Fg);
			//update_velocity(&particles_new[i],TIME);
			update_position(&particles_new[i],TIME);
			/*printf("%f %f %f %f %f\n",particles_old[i].position.elements[0],particles_old[i].position.elements[1],particles_old[i].position.elements[2],
			                          particles_old[i].velocity.elements[0],particles_old[i].velocity.elements[2]);*/
			printf("%f %f %f %f %f %f %f %f %f\n",Fg.elements[0],Fg.elements[1],Fg.elements[2],
									particles_new[i].acceleration.elements[0],particles_new[i].acceleration.elements[1],particles_new[i].acceleration.elements[2],
									particles_new[i].velocity.elements[0],particles_new[i].velocity.elements[1],particles_new[i].velocity.elements[2]);
		}
		printf("\n");

		sprintf(filename,"%03i.png",file_cnt);
		write_file(filename);
		file_cnt++;

		memcpy(particles_old,particles_new,sizeof(particle)*NUM_PARTICLES);
	}

	return(0);
}

void write_file(char* filename) {
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
	x_pos = (int)((abs(MIN_X_IMG) + center_of_mass.position.elements[0]) / scale_x);
	y_pos = (int)((abs(MIN_Y_IMG) + center_of_mass.position.elements[2]) / scale_y);

	png_byte* row = row_pointers[y_pos];
	png_byte* ptr = &(row[x_pos*4]);
	ptr[0] = 255; ptr[1] = 255; ptr[2] = 255; ptr[3] = 255;

	for(i=0; i < NUM_PARTICLES; i++) {
		if((particles_old[i].position.elements[0] > MIN_X_IMG && particles_old[i].position.elements[0] < MAX_X_IMG) &&
		   (particles_old[i].position.elements[2] > MIN_Y_IMG && particles_old[i].position.elements[2] < MAX_Y_IMG)) {
			x_pos = (int)((abs(MIN_X_IMG) + particles_old[i].position.elements[0]) / scale_x);
			y_pos = (int)((abs(MIN_Y_IMG) + particles_old[i].position.elements[2]) / scale_y);

			png_byte* row = row_pointers[y_pos];
			png_byte* ptr = &(row[x_pos*4]);
			ptr[0] = 255; ptr[1] = 0; ptr[2] = 0; ptr[3] = 255;
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
}
