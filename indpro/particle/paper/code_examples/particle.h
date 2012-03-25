#include <stdlib.h>
#include <math.h>

#ifndef __PARTICLE_H__
#define __PARTICLE_H__

#define MAX_PARMS 25

#define TRUE 1
#define FALSE 0

typedef char BOOL;

typedef struct
{
	double x,y,z;                     /* x,y,z position vector */
	double (*x_pos_fnt)(struct particle*, float); /* x position function */
	double (*y_pos_fnt)(struct particle*, float); /* y position function */
	double (*z_pos_fnt)(struct particle*, float); /* z position function */
	float parm_list[MAX_PARMS];                   /* parameter list */
} particle;

BOOL init_particle(particle* p, double x, double y, double z);
BOOL set_x_pos_fnt(particle* p, double (*x_pos_fnt)(particle*, float));
BOOL set_y_pos_fnt(particle* p, double (*y_pos_fnt)(particle*, float));
BOOL set_z_pos_fnt(particle* p, double (*z_pos_fnt)(particle*, float));

void update_particle(particle* p, float t);

#endif
