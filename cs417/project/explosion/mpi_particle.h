#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <memory.h>

#ifndef __PARTICLE_H__
#define __PARTICLE_H__

typedef struct {
	float mass;
	float position[3];
	float acceleration[3];
	float velocity[3];
} particle;

void particle_init(particle* p) {
	memset(&p->position,0,sizeof(float)*3);
	memset(&p->acceleration,0,sizeof(float)*3);
	memset(&p->velocity,0,sizeof(float)*3);
	p->mass = 0;
}

#endif
