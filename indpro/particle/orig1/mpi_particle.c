#include <stdlib.h>
#include <math.h>
#include "mpi_particle.h"

BOOL init_particle(particle* p, double x, double y, double z) {
	if(p == NULL) {
		return(FALSE);
	}

	p->x = x;
	p->y = y;
	p->z = z;
	memset(&p->parm_list,0,sizeof(parm_type)*MAX_PARMS);

	return(TRUE);
}

BOOL init_particle_engine() {
	pos_fnt[POS_FNT_X] = NULL;
	pos_fnt[POS_FNT_Y] = NULL;
	pos_fnt[POS_FNT_Z] = NULL;

	return(TRUE);
}

BOOL set_x_pos_fnt(particle* p, double (*x_pos_fnt)(particle*, float)) {
	if(p == NULL || x_pos_fnt == NULL) {
		return(FALSE);
	}

	pos_fnt[POS_FNT_X] = x_pos_fnt;
	return(TRUE);
}

BOOL set_y_pos_fnt(particle* p, double (*y_pos_fnt)(particle*, float)) {
	if(p == NULL || y_pos_fnt == NULL) {
		return(FALSE);
	}

	pos_fnt[POS_FNT_Y] = y_pos_fnt;
	return(TRUE);
}

BOOL set_z_pos_fnt(particle* p, double (*z_pos_fnt)(particle*, float)) {
	if(p == NULL || z_pos_fnt == NULL) {
		return(FALSE);
	}

	pos_fnt[POS_FNT_Z] = z_pos_fnt;
	return(TRUE);
}

void update_particle(particle* p, float t) {
	if(pos_fnt[POS_FNT_X] == NULL || pos_fnt[POS_FNT_Y] == NULL || pos_fnt[POS_FNT_Z] == NULL) {
		return;
	}

	p->x = pos_fnt[POS_FNT_X](p, t);
	p->y = pos_fnt[POS_FNT_Y](p, t);
	p->z = pos_fnt[POS_FNT_Z](p, t);
}
