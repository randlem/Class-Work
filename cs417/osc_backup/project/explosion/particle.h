#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <memory.h>
#include "vector.h"

#ifndef __PARTICLE_H__
#define __PARTICLE_H__

typedef struct {
	float mass;
	float alpha;
	vector position;
	vector acceleration;
	vector velocity;
} particle;

void particle_init(particle* p) {
	vector_init(&p->position);
	vector_init(&p->acceleration);
	vector_init(&p->velocity);
	p->mass = 0;
}

void update_position(particle* p, int t) {
	int i;

	for(i=0; i < CARDNALITY; i++)
		p->position.elements[i] = (p->velocity.elements[i] * t) + (.5 * t * t * p->acceleration.elements[i]);
}

#endif
