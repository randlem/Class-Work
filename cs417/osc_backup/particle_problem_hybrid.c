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
#include <mpi.h>
#include <omp.h>
#include "vector.h"
#include "mpi_particle.h"

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

void parent();
void child();

int num_particles;
int initial_height;
vector initial_velocity;
particle* particles;
int id,num_nodes;

MPI_Datatype particle_type;
int blockcounts[4] = {1,3,3,3};
MPI_Datatype types[4] = {MPI_FLOAT,MPI_FLOAT,MPI_FLOAT,MPI_FLOAT};
MPI_Aint displs[4];

int main(int argc, char* argv[]) {
    int i;
	MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    MPI_Comm_size(MPI_COMM_WORLD,&num_nodes);

	omp_set_num_threads(2);

	srand((int)time(NULL));

	initial_height = 0;
	initial_velocity.elements[0] = 0;
	initial_velocity.elements[1] = 0;
	initial_velocity.elements[2] = 0;

	if(id == 0) {
		if(argc > 1) {
			if(argc == 3) {
				initial_height = atoi(argv[2]);
				initial_velocity.elements[0] = 0;
				initial_velocity.elements[1] = 0;
				initial_velocity.elements[2] = 0;
			} else if(argc == 6) {
				initial_height = atoi(argv[2]);
				initial_velocity.elements[0] = atof(argv[3]);
				initial_velocity.elements[1] = atof(argv[4]);
				initial_velocity.elements[2] = atof(argv[5]);
			} else if(argc != 2){
				fprintf(stderr,"Please specify either an initial heigth or an inital height and velocity (in vector components).\n");
				return(1);
			}
		} else {
			fprintf(stderr,"Please specify a number of particles.\n");
			return(1);
		}

		num_particles = atoi(argv[1]);

	}

	MPI_Bcast(&num_particles,1,MPI_INT,0,MPI_COMM_WORLD);

	particles = (particle*)malloc(sizeof(particle)*num_particles);
	memset(particles,0,sizeof(particle)*num_particles);

	MPI_Address(&particles[0].mass,&displs[0]);
	MPI_Address(&particles[0].position,&displs[1]);
	MPI_Address(&particles[0].acceleration,&displs[2]);
	MPI_Address(&particles[0].velocity,&displs[3]);

	for(i=0; i < 4; i++)
		displs[i] -= displs[0];

	MPI_Type_struct(4,blockcounts,displs,types,&particle_type);
	MPI_Type_commit(&particle_type);

	if(id == 0) {
		parent();
	} else {
		child();
	}

	free(particles);

	MPI_Finalize();

	return(0);
}

void parent() {
	int i,j,k;
	int num_frames,partitions[num_nodes],baseline,extra,offset;
	float vel,ke,theta,alpha;
	int flag=0;
    MPI_Status status;

	#pragma omp parallel shared(particles,num_particles,initial_height) private(i,ke,theta,alpha,vel)
	{
		#pragma omp for nowait
		for(i=0; i < num_particles; i++) {
			particle_init(&particles[i]);
			particles[i].acceleration[1] = GRAVITY;
			particles[i].mass = RAND_RANGE(1,100);
			particles[i].position[1] = initial_height;

			ke = RAND_RANGE(BASE_KE,BASE_KE+500);
			theta = FP_RAND_RANGE(0,PI);

			if(initial_height > 0) {
				alpha = FP_RAND_RANGE(0,TWO_PI);
			} else {
				alpha = FP_RAND_RANGE(0,PI);
			}

			vel = sqrt((2 * ke)/particles[i].mass);

			particles[i].velocity[0] = (vel * cos(alpha) * sin(theta)) + initial_velocity.elements[0];
			particles[i].velocity[1] = (vel * sin(alpha) * sin(theta)) + initial_velocity.elements[1];
			particles[i].velocity[2] = (vel * cos(theta)) + initial_velocity.elements[2];
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Bcast(particles,num_particles*sizeof(particle),MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);

	num_frames = MAX_TIME/TIME_INC;
	baseline = (int)((float)num_frames/(float)(num_nodes-1));
	extra = num_frames%(num_nodes-1);
	memset(partitions,0,sizeof(int)*num_nodes);
	for(i=1; i < num_nodes; i++) {
		partitions[i] = baseline;
		if(extra > 0) {
			partitions[i]++;
			extra--;
		}
	}

	offset=0;
	for(i=1; i < num_nodes; i++) {
		MPI_Send(&partitions[i],1,MPI_INT,i,0,MPI_COMM_WORLD);
		MPI_Send(&offset,1,MPI_INT,i,0,MPI_COMM_WORLD);
		offset+=partitions[i];
	}

	MPI_Barrier(MPI_COMM_WORLD);

	for(i=1; i < num_nodes; i++) {
		MPI_Send(&flag,1,MPI_INT,i,0,MPI_COMM_WORLD);
		MPI_Recv(&flag,1,MPI_INT,i,0,MPI_COMM_WORLD,&status);
		/*usleep(40);*/
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

void child() {
	int i,t;
	int partition,start;
	int flag=0;
	char filename[80];
	char buffer[80];
	MPI_File fh;
	MPI_Status status;
	particle** states;

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Bcast(particles,num_particles*sizeof(particle),MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);

	MPI_Recv(&partition,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
	MPI_Recv(&start,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);

	states = (particle**)malloc(sizeof(particle*)*partition);
	for(i=0; i < partition; i++) {
		states[i] = (particle*)malloc(sizeof(particle)*num_particles);
	}

	if(start != 0) {
		for(i=0; i < num_particles; i++) {
			if(particles[i].velocity[0] != 0.0F && particles[i].velocity[1] != 0.0F && particles[i].velocity[2] != 0.0F) {
				particles[i].position[0] += (particles[i].velocity[0] * TIME_INC * start) + (particles[i].acceleration[0] * SQUARE(TIME_INC * start) * 0.5F);
				particles[i].position[1] += (particles[i].velocity[1] * TIME_INC * start) + (particles[i].acceleration[1] * SQUARE(TIME_INC * start) * 0.5F);
				particles[i].position[2] += (particles[i].velocity[2] * TIME_INC * start) + (particles[i].acceleration[2] * SQUARE(TIME_INC * start) * 0.5F);

				particles[i].velocity[0] += (particles[i].acceleration[0] * TIME_INC * start);
				particles[i].velocity[1] += (particles[i].acceleration[1] * TIME_INC * start);
				particles[i].velocity[2] += (particles[i].acceleration[2] * TIME_INC * start);
			}
			if(particles[i].position[1] < 0.0F) {
				particles[i].position[1] = 0.0F;
				particles[i].velocity[0] = 0.0F;
				particles[i].velocity[1] = 0.0F;
				particles[i].velocity[2] = 0.0F;
			}
			memcpy(&states[0][i],&particles[i],sizeof(particle));
		}
	}

	for(t=1; t < partition; t++) {
		#pragma omp parallel shared(particles,states,num_particles,t) private(i)
		{
			#pragma omp for nowait
			for(i=0; i < num_particles; i++) {
				if(particles[i].velocity[0] != 0.0F && particles[i].velocity[1] != 0.0F && particles[i].velocity[2] != 0.0F) {
					particles[i].position[0] += (particles[i].velocity[0] * TIME_INC) + (particles[i].acceleration[0] * SQUARE(TIME_INC) * 0.5F);
					particles[i].position[1] += (particles[i].velocity[1] * TIME_INC) + (particles[i].acceleration[1] * SQUARE(TIME_INC) * 0.5F);
					particles[i].position[2] += (particles[i].velocity[2] * TIME_INC) + (particles[i].acceleration[2] * SQUARE(TIME_INC) * 0.5F);

					particles[i].velocity[0] += (particles[i].acceleration[0] * TIME_INC);
					particles[i].velocity[1] += (particles[i].acceleration[1] * TIME_INC);
					particles[i].velocity[2] += (particles[i].acceleration[2] * TIME_INC);
				}
				if(particles[i].position[1] < 0.0F) {
					particles[i].position[1] = 0.0F;
					particles[i].velocity[0] = 0.0F;
					particles[i].velocity[1] = 0.0F;
					particles[i].velocity[2] = 0.0F;
				}
				memcpy(&states[t][i],&particles[i],sizeof(particle));
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Recv(&flag,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
	for(t=0; t < partition; t++) {
		/*printf("%i\n",num_particles);*/
		for(i=0; i < num_particles; i++) {
			/*printf("%i %f %f %f\n",start+t,states[t][i].position[0],states[t][i].position[1],states[t][i].position[2]);*/
		}/*printf("\n");*/
	}
	MPI_Send(&flag,1,MPI_INT,0,0,MPI_COMM_WORLD);

	for(i=0; i < partition; i++) {
		free(states[i]);
	}
	free(states);

	MPI_Barrier(MPI_COMM_WORLD);
}
