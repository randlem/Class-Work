#include <stdlib.h>
#include <math.h>
#include "particle.h"

BOOL init_particle(particle* p, double x, double y, double z) {
	if(p == NULL) {
		return(FALSE);
	}

	p->x = x;
	p->y = y;
	p->z = z;
	p->x_pos_fnt = NULL;
	p->y_pos_fnt = NULL;
	p->z_pos_fnt = NULL;

	return(TRUE);
}

BOOL set_x_pos_fnt(particle* p, double (*x_pos_fnt)(particle*, float)) {
	if(p == NULL || x_pos_fnt == NULL) {
		return(FALSE);
	}

	p->x_pos_fnt = x_pos_fnt;
	return(TRUE);
}

BOOL set_y_pos_fnt(particle* p, double (*y_pos_fnt)(particle*, float)) {
	if(p == NULL || y_pos_fnt == NULL) {
		return(FALSE);
	}

	p->y_pos_fnt = y_pos_fnt;
	return(TRUE);
}

BOOL set_z_pos_fnt(particle* p, double (*z_pos_fnt)(particle*, float)) {
	if(p == NULL || z_pos_fnt == NULL) {
		return(FALSE);
	}

	p->z_pos_fnt = z_pos_fnt;
	return(TRUE);
}

void update_particle(particle* p, float t) {
	if(p->x_pos_fnt == NULL || p->y_pos_fnt == NULL || p->z_pos_fnt == NULL) {
		return;
	}

	p->x = p->x_pos_fnt(p, t);
	p->y = p->y_pos_fnt(p, t);
	p->z = p->z_pos_fnt(p, t);
}
