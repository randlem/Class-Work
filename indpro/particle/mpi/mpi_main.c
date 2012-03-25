/************************************************
 * MPI Datatypes test
 ************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <mpi.h>

#include "mpi_particle.h"
#include "png_writer.h"

/* macros */
#define RAND_RANGE(a,b) ((a)+(rand()%((b)-(a)+1)))
#define FP_RAND_RANGE(a,b) ((a)+((rand()/(float)RAND_MAX)*(b)))
#define SQUARE(a) ((a)*(a))
#define RAD_TO_DEG(a) (((a)*180.0F)/PI)

/* defines */
#define NUM_PARTICLES     100

#define TOTAL_PARTICLE_KE 100000 /* J */

#define GRAVITY -9.8F /* m/s^2 */
#define PI 3.1415926F
#define TWO_PI 6.2831852F
#define HALF_PI 1.5707963F
#define NEG_HALF_PI -1.5707963F

#define NUM_PARMS 7
#define MASS 0
#define X_V0 1
#define Y_V0 2
#define Z_V0 3
#define X_A  4
#define Y_A  5
#define Z_A  6

double position_x(particle* p, float t) {
	return((p->parm_list[X_V0] * t) + (p->parm_list[X_A] * .5F * t * t));
}

double position_y(particle* p, float t) {
	return((p->parm_list[Y_V0] * t) + (p->parm_list[Y_A] * .5F * t * t));
}

double position_z(particle* p, float t) {
	return((p->parm_list[Z_V0] * t) + (p->parm_list[Z_A] * .5F * t * t));
}

int main(int argc, char* argv[]) {
	particle p_list[NUM_PARTICLES];
	png_file png;
	float vel,alpha,theta;
	float t;
	int i,j;
	int* partitions = NULL;
	int node_partition;

	int num_nodes,id;
	MPI_Status status;

	/* init MPI */
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    MPI_Comm_size(MPI_COMM_WORLD,&num_nodes);

	memset(&p_list,0,sizeof(particle)*NUM_PARTICLES);

	init_particle_engine();
	set_x_pos_fnt(&p_list[i],&position_x);
	set_y_pos_fnt(&p_list[i],&position_y);
	set_z_pos_fnt(&p_list[i],&position_z);

	/* do the parent thread stuff */
	if(id == 0) {

		/* init the particle list */
		for(i=0; i < NUM_PARTICLES; i++) {
			init_particle(&p_list[i],0,0,0);

			p_list[i].parm_list[MASS] = FP_RAND_RANGE(1.0F,10.0F);       /* kg */

			alpha = FP_RAND_RANGE(0,PI);
			theta = FP_RAND_RANGE(0,PI);
			vel = sqrt((2 * TOTAL_PARTICLE_KE)/p_list[i].parm_list[MASS]);

			p_list[i].parm_list[X_V0] = (vel * cos(alpha) * sin(theta)); /* m/s */
			p_list[i].parm_list[Y_V0] = (vel * sin(alpha) * sin(theta)); /* m/s */
			p_list[i].parm_list[Z_V0] = (vel * cos(theta));              /* m/s */
			p_list[i].parm_list[X_A] = 0;                                /* m/s^2 */
			p_list[i].parm_list[Y_A] = GRAVITY;                          /* m/s^2 */
			p_list[i].parm_list[Z_A] = 0;                                /* m/s^2 */

		}

		/* open the png file */
		open_file(&png,"test.png",1024,1024,0,2000,-2000,0);

		/* dynamically allocate the number of partitions I'll need */
		partitions = (int*)malloc(sizeof(int)*(num_nodes-1));
	}

	/* send NUM_PARTICLES/num_nodes particles to each child process */
	if(id == 0) {
		int extra = NUM_PARTICLES % (num_nodes-1);
		int std_partition = NUM_PARTICLES / (num_nodes-1);
		int offset = 0;

		for(i=0; i < num_nodes - 1; i++) {
			partitions[i] = std_partition;
			if(extra != 0) {
				partitions[i]++;
				extra--;
			}
		}

		for(i=1; i < num_nodes; i++) {
			MPI_Send(&partitions[i-1],1,MPI_INT,i,0,MPI_COMM_WORLD);
			for(j=0; j < partitions[i-1]; j++) {
				MPI_Send(&p_list[offset+j],sizeof(particle),MPI_BYTE,i,0,MPI_COMM_WORLD);
			}
			offset += partitions[i];
		}
	} else {
		MPI_Recv(&node_partition,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		for(j=0; j < node_partition; j++) {
			MPI_Recv(&p_list[j],sizeof(particle),MPI_BYTE,0,0,MPI_COMM_WORLD,&status);
			printf("%f %f %f\n",p_list[j].parm_list[X_V0],p_list[j].parm_list[Y_V0],p_list[j].parm_list[Z_V0]);
		}
	}

	/* do some work */
	if(id != 0) {
		for(i=0; i < node_partition; i++) {
			update_particle(&p_list[i],t);
			/*printf("%F %F %F\n",p_list[i].x,p_list[i].y,p_list[i].z);*/
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* do some parent stuff */
	if(id == 0) {
		/* write out the png file */
		write_file(&png);

		/* close the png file */
		close_file(&png);

		/* free mai memory */
		free(partitions);
	}

	/* shutdown MPI */
	MPI_Finalize();

	return(0);
}
