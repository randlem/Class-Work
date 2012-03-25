#include <stdlib.h>
#include <math.h>

#ifndef __MPI_PARTICLE_H__
#define __MPI_PARTICLE_H__

/* position function pointer hash stuff */
#define NUM_POS_FNT 3
#define POS_FNT_X 0
#define POS_FNT_Y 1
#define POS_FNT_Z 2

typedef double (*pos_fnt_ptr) (struct particle*, float);

/* bool type stuff */
#define TRUE 1
#define FALSE 0

typedef char BOOL;

/* particle type declare */
#define MAX_PARMS 10

typedef float parm_type;
typedef struct
{
	double x,y,z;                   /* x,y,z position vector */
	parm_type parm_list[MAX_PARMS]; /* parameter list */
} particle;

/* global array for positing function pointers */
pos_fnt_ptr pos_fnt[NUM_POS_FNT];

BOOL init_particle_engine();
BOOL init_particle(particle* p, double x, double y, double z);
BOOL set_x_pos_fnt(particle* p, double (*x_pos_fnt)(particle*, float));
BOOL set_y_pos_fnt(particle* p, double (*y_pos_fnt)(particle*, float));
BOOL set_z_pos_fnt(particle* p, double (*z_pos_fnt)(particle*, float));

void update_particle(particle* p, float t);

#endif
