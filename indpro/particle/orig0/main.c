#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include "particle.h"
#include "png_writer.h"

#define NUM_PARTICLES 1

#define GRAVITY -9.8F /* (m/s)^2 */

#define NUM_PARMS 4
#define MASS 0
#define X_V0 1
#define Y_V0 2
#define Z_V0 3

double position_x(particle* p, float t) {
	return(p->parm_list[X_V0]*t);
}

double position_y(particle* p, float t) {
	return((p->parm_list[Y_V0] * t) + (GRAVITY * .5F * t * t));
}

double position_z(particle* p, float t) {
	return(0);
}

int main(int argc, char* argv[]) {
	particle p_list[NUM_PARTICLES];
	png_file png;
	float t;
	int i;

	/* init the particle list */
	for(i=0; i < NUM_PARTICLES; i++) {
		init_particle(&p_list[i],0,0,0);
		set_x_pos_fnt(&p_list[i],&position_x);
		set_y_pos_fnt(&p_list[i],&position_y);
		set_z_pos_fnt(&p_list[i],&position_z);
		p_list[i].parm_list[MASS] = 1.0F;  /* kg */
		p_list[i].parm_list[X_V0] = 15.0F; /* m/s */
		p_list[i].parm_list[Y_V0] = 0.0F;  /* m/s */
		p_list[i].parm_list[Z_V0] = 0.0F;  /* m/s */
	}

	/* open the png file */
	open_file(&png,"test.png",1024,1024,0,2000,-2000,0);

	/* do some work */
	for(t=0.0F; t < 20.0F; t+=0.05F) {
		for(i=0; i < NUM_PARTICLES; i++) {
			update_particle(&p_list[i],t);
			/*printf("%F %F %F\n",p_list[i].x,p_list[i].y,p_list[i].z);*/
			plot_point(&png,(int)p_list[i].x,(int)p_list[i].y,255,255,255);
		}
	}

	//plot_point(&png,0,0,255,255,255);

	/* write out the png file */
	write_file(&png);

	/* close the png file */
	close_file(&png);

	return(0);
}
