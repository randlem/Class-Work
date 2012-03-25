#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <memory.h>

#ifndef __VECTOR_H__
#define __VECTOR_H__

#define CARDNALITY 3

typedef float vector_element;

typedef struct {
	vector_element elements[CARDNALITY];  /* vector elements */
} vector3d;

typedef vector3d vector;

void vector_init(vector* v) {
	memset(v->elements,0,sizeof(vector_element)*CARDNALITY);
}

float vector_length(vector* v) {
	float sum = 0; int i;

	for(i=0; i < CARDNALITY; i++)
		sum += v->elements[i]*v->elements[i];

	return(sqrt(sum));
}

void normalize(vector* v) {
	float len = vector_length(v);
	int i;

	for(i=0; i < CARDNALITY; i++)
		v->elements[i] = v->elements[i]/len;
}

void vector_add(vector* v, vector* u, vector* r) {
	int i;

	for(i=0; i < CARDNALITY; i++)
		r->elements[i] = v->elements[i] + u->elements[i];
}

float vector_dot(vector* v, vector* u) {
	int i; float dot=0;

	for(i=0; i < CARDNALITY; i++)
		dot += v->elements[i] * u->elements[i];

	return(dot);
}

void vector_scalar(vector* v, float scalar) {
	int i;

	for(i=0; i < CARDNALITY; i++)
		v->elements[i] *= scalar;

}

#endif
