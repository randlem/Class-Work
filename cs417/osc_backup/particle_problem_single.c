/******************************************************************************
* Particle Explosing Sim - Single Processor Version
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

#define GRAVITY -9.8F /* m/s^2 */
#define PI 3.1415926F
#define TWO_PI 6.2831852F
#define HALF_PI 1.5707963F
#define NEG_HALF_PI -1.5707963F

#define MAX_TIME 10
#define TIME_INC 0.03F
#define BASE_KE 3000

#define RAND_RANGE(a,b) ((a)+(rand()%((b)-(a)+1)))
#define FP_RAND_RANGE(a,b) ((a)+((rand()/(float)RAND_MAX)*(b)))
#define SQUARE(a) ((a)*(a))
#define RAD_TO_DEG(a) (((a)*180.0F)/PI)

void write_file_fake(char* filename);

int num_particles;

particle* particles;
vector initial_velocity;
int initial_height;

int main(int argc, char* argv[]) {
	int i;
	float vel,   /* initial velocity vector */
	      ke,    /* initial imparted kinetic energy */
	      theta, /* theta angle, elevation angle off the x-y plane */
		  alpha; /* alpha angle, azmuth on the x-y plane */
	int file_cnt;
	float t;
	char filename[80];

	/* do a bit of initilization for the cmd line stuff */
	initial_height = 0;
	initial_velocity.elements[0] = 0;
	initial_velocity.elements[1] = 0;
	initial_velocity.elements[2] = 0;

	/* validate the cmd line stuff */
	if(argc > 1) {
		if(argc == 3) {
			/* just a initial height */
			initial_height = atoi(argv[2]);
			initial_velocity.elements[0] = 0;
			initial_velocity.elements[1] = 0;
			initial_velocity.elements[2] = 0;
		} else if(argc == 6) {
			/* initial height and initial velocity */
			initial_height = atoi(argv[2]);
			initial_velocity.elements[0] = atof(argv[3]);
			initial_velocity.elements[1] = atof(argv[4]);
			initial_velocity.elements[2] = atof(argv[5]);
		} else if(argc != 2){
			/* otherwise it's an error and they need a message */
			fprintf(stderr,"Please specify either an initial heigth or an inital height and velocity (in vector components).\n");
			return(1);
		}
	} else {
		/* otherwise it's an error and they need a message */
		fprintf(stderr,"Please specify a number of particles.\n");
		return(1);
	}

	num_particles = atoi(argv[1]);

	/* seed the random number generator */
	srand((unsigned)time(NULL));

	/* allocate the particles */
	particles = (particle*)malloc(sizeof(particle)*num_particles);

	/* do the easy initlization */
	for(i=0; i < num_particles; i++) {
		particle_init(&particles[i]);
		particles[i].acceleration.elements[1] = GRAVITY;
		particles[i].mass = RAND_RANGE(1,100);
		particles[i].position.elements[1] = initial_height;
	}

	/* set the initial velocity of all the particles */
	for(i=0; i < num_particles; i++) {
		ke = RAND_RANGE(BASE_KE,BASE_KE+500); /* generate a random inital kinetic energy */
		theta = FP_RAND_RANGE(0,PI);          /* generate a random initial theta */

		/* create the approiate alpha */
		if(initial_height > 0) {
			alpha = FP_RAND_RANGE(0,TWO_PI);
		} else {
			alpha = FP_RAND_RANGE(0,PI);
		}

		/* caculate the initial velocity */
		vel = sqrt((2 * ke)/particles[i].mass);

		/* resolve the initial velocity into x-y-z components */
		particles[i].velocity.elements[0] = (vel * cos(alpha) * sin(theta)) + initial_velocity.elements[0]; /* x component */
		particles[i].velocity.elements[1] = (vel * sin(alpha) * sin(theta)) + initial_velocity.elements[1]; /* y component */
		particles[i].velocity.elements[2] = (vel * cos(theta)) + initial_velocity.elements[1];              /* z component */
	}

	/* init the counters */
	file_cnt = 0; t = 0;

	/* do a single time element once */
	for(i=0; i < num_particles; i++) {
		particles[i].position.elements[0] += (particles[i].velocity.elements[0] * TIME_INC) + (particles[i].acceleration.elements[0] * SQUARE(TIME_INC) * 0.5F);
		particles[i].position.elements[1] += (particles[i].velocity.elements[1] * TIME_INC) + (particles[i].acceleration.elements[1] * SQUARE(TIME_INC) * 0.5F);
		particles[i].position.elements[2] += (particles[i].velocity.elements[2] * TIME_INC) + (particles[i].acceleration.elements[2] * SQUARE(TIME_INC) * 0.5F);

		particles[i].velocity.elements[0] += (particles[i].acceleration.elements[0] * TIME_INC);
		particles[i].velocity.elements[1] += (particles[i].acceleration.elements[1] * TIME_INC);
		particles[i].velocity.elements[2] += (particles[i].acceleration.elements[2] * TIME_INC);
	}
	for(i=0; i < num_particles; i++) {
		if(particles[i].position.elements[1] < 0.0F)
			particles[i].position.elements[1] = 0.0F;
	}
	sprintf(filename,"%03i.png",file_cnt);
	write_file_fake(filename);
	file_cnt++;
	t += TIME_INC;

	/* loop till i hit the max time */
	while(t < MAX_TIME) {
		for(i=0; i < num_particles; i++) {
			if(particles[i].position.elements[1] > 0.0F) {
				particles[i].position.elements[0] += (particles[i].velocity.elements[0] * TIME_INC) + (particles[i].acceleration.elements[0] * SQUARE(TIME_INC) * 0.5F);
				particles[i].position.elements[1] += (particles[i].velocity.elements[1] * TIME_INC) + (particles[i].acceleration.elements[1] * SQUARE(TIME_INC) * 0.5F);
				particles[i].position.elements[2] += (particles[i].velocity.elements[2] * TIME_INC) + (particles[i].acceleration.elements[2] * SQUARE(TIME_INC) * 0.5F);

				particles[i].velocity.elements[0] += (particles[i].acceleration.elements[0] * TIME_INC);
				particles[i].velocity.elements[1] += (particles[i].acceleration.elements[1] * TIME_INC);
				particles[i].velocity.elements[2] += (particles[i].acceleration.elements[2] * TIME_INC);
			}
		}
		for(i=0; i < num_particles; i++) {
			if(particles[i].position.elements[1] < 0.0F)
				particles[i].position.elements[1] = 0.0F;
		}
		sprintf(filename,"%03i.png",file_cnt);
		write_file_fake(filename);
		file_cnt++;
		t += TIME_INC;
	}

	/* free the particles */
	free(particles);

	return(0);
}

void write_file_fake(char* filename) {
	int i;

	printf("%i\n",num_particles);
	for(i=0; i < num_particles; i++) {
		printf("%f %f %f\n",particles[i].position.elements[0],particles[i].position.elements[1],particles[i].position.elements[2]);
	}
	printf("\n");
}
