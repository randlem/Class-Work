#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "vector.h"

#ifndef __PARTICLE_H__
#define __PARTICLE_H__

typedef struct {
	int id;              /* particle id */
	double mass;         /* mass in kg */
	vector velocity;     /* velocity vector */
	vector acceleration; /* acceleration vector */
	vector position;     /* position vector */
} particle;

float particle_distance(particle* p0,particle* p1) {
	float r=0,t=0; int i;

	for(i=0; i < CARDNALITY; i++) {
		t = abs(p0->position.elements[i] - p1->position.elements[i]);
		r += t*t;
	}

	return(abs(sqrt(r)));
}

void update_position(particle* p,float t) {
	int i;

	for(i=0; i < CARDNALITY; i++)
		p->position.elements[i] += (p->velocity.elements[i] * t) + (p->acceleration.elements[i] * t * t * .5);
}

void update_velocity(particle* p,float t) {
	int i;

	for(i=0; i < CARDNALITY; i++)
		p->velocity.elements[i] += (p->velocity.elements[i] * t) + (p->acceleration.elements[i] * t * t * .5);
}

void update_acceleration(particle* p,vector* F) {
	int i;

	for(i=0; i < CARDNALITY; i++)
		p->acceleration.elements[i] = (float)((float)F->elements[i] / (float)p->mass);
}

#endif
