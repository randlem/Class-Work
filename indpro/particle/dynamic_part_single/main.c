#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>

#include "mpi_particle.h"
#include "png_writer.h"

/* macros */
#define RAND_RANGE(a,b) ((a)+(rand()%((b)-(a)+1)))
#define FP_RAND_RANGE(a,b) ((a)+((rand()/(float)RAND_MAX)*(b)))
#define SQUARE(a) ((a)*(a))
#define RAD_TO_DEG(a) (((a)*180.0F)/PI)

/* defines */
#define NUM_PARTICLES     100

#define TOTAL_PARTICLE_KE 2500 /* J */

#define GRAVITY -9.8F /* m/s^2 */
#define PI 3.1415926F
#define TWO_PI 6.2831852F
#define HALF_PI 1.5707963F
#define NEG_HALF_PI -1.5707963F

#define NUM_PARMS 10
#define MASS 0
#define X_V0 1
#define Y_V0 2
#define Z_V0 3
#define X_A  4
#define Y_A  5
#define Z_A  6
#define X0   7
#define Y0   8
#define Z0   9

typedef struct {
	int id;
	float x0,x1;
	float y0,y1;
} path2d;

double position_x(particle* p, float t) {
	return((p->parm_list[X_V0] * t) + (p->parm_list[X_A] * .5F * t * t));
}

double position_y(particle* p, float t) {
	return((p->parm_list[Y_V0] * t) + (p->parm_list[Y_A] * .5F * t * t));
}

double position_z(particle* p, float t) {
	return((p->parm_list[Z_V0] * t) + (p->parm_list[Z_A] * .5F * t * t));
}

void sort_particles(path2d* path_list, int size_path_list, int* interacting, int* non_interacting);

void calc_resting(particle* p_list, path2d* resting);
BOOL calc_intersect(float Ax,float Bx,float Cx,float Dx,float Ay,float By,float Cy,float Dy);

int main(int argc, char* argv[]) {
	particle p_list_1[NUM_PARTICLES];
	particle p_list_2[NUM_PARTICLES];
	int interacting_list[NUM_PARTICLES*2];
	int non_interacting_list[NUM_PARTICLES*2];
	path2d precompute_path_list[NUM_PARTICLES * 2];
	png_file png,png1;
	float vel,alpha,theta;
	float t;
	int i,j;

	memset(&p_list_1,0,sizeof(particle)*NUM_PARTICLES);
	memset(&p_list_2,0,sizeof(particle)*NUM_PARTICLES);

	init_particle_engine();
	set_x_pos_fnt(&position_x);
	set_y_pos_fnt(&position_y);
	set_z_pos_fnt(&position_z);

	open_file(&png,"x-z_start_pos.png",1024,1024,-200,200,-200,200);
	open_file(&png1,"y-z_start_pos.png",1024,1024,-200,200,-200,200);

	for(i=0; i < NUM_PARTICLES; i++) {
		alpha = FP_RAND_RANGE(0,PI);
		theta = -FP_RAND_RANGE(0,PI);

		init_particle(&p_list_1[i],-100 + (10.0F * cos(alpha) * sin(theta)),0 + (10.0F * sin(alpha) * sin(theta)),100 + (10.0F * cos(theta)));
		plot_point(&png,p_list_1[i].x,p_list_1[i].y,255,255,255);

		p_list_1[i].id              = i;
		p_list_1[i].parm_list[MASS] = FP_RAND_RANGE(1.0F,10.0F);       /* kg */

		vel = sqrt((2 * TOTAL_PARTICLE_KE)/p_list_1[i].parm_list[MASS]);

		p_list_1[i].parm_list[X0] = p_list_1[i].x;                     /* m */
		p_list_1[i].parm_list[Y0] = p_list_1[i].y;                     /* m */
		p_list_1[i].parm_list[Z0] = p_list_1[i].z;                     /* m */
		p_list_1[i].parm_list[X_V0] = (vel * cos(alpha) * sin(theta)); /* m/s */
		p_list_1[i].parm_list[Y_V0] = (vel * sin(alpha) * sin(theta)); /* m/s */
		p_list_1[i].parm_list[Z_V0] = (vel * cos(theta));              /* m/s */
		p_list_1[i].parm_list[X_A] = 0;                                /* m/s^2 */
		p_list_1[i].parm_list[Y_A] = GRAVITY;                          /* m/s^2 */
		p_list_1[i].parm_list[Z_A] = 0;                                /* m/s^2 */

		alpha = FP_RAND_RANGE(0,PI);
		theta = -FP_RAND_RANGE(0,PI);

		init_particle(&p_list_2[i],100 + (10.0F * cos(alpha) * sin(theta)), 0 + (10.0F * sin(alpha) * sin(theta)), -100 + (10.0F * cos(theta)));
		plot_point(&png,p_list_2[i].x,p_list_2[i].y,255,255,255);

		p_list_2[i].id              = i+NUM_PARTICLES;
		p_list_2[i].parm_list[MASS] = FP_RAND_RANGE(1.0F,10.0F);       /* kg */

		vel = sqrt((2 * TOTAL_PARTICLE_KE)/p_list_2[i].parm_list[MASS]);

		p_list_2[i].parm_list[X0] = p_list_2[i].x;                     /* m */
		p_list_2[i].parm_list[Y0] = p_list_2[i].y;                     /* m */
		p_list_2[i].parm_list[Z0] = p_list_2[i].z;                     /* m */
		p_list_2[i].parm_list[X_V0] = (vel * cos(alpha) * sin(theta)); /* m/s */
		p_list_2[i].parm_list[Y_V0] = (vel * sin(alpha) * sin(theta)); /* m/s */
		p_list_2[i].parm_list[Z_V0] = (vel * cos(theta));              /* m/s */
		p_list_2[i].parm_list[X_A] = 0;                                /* m/s^2 */
		p_list_2[i].parm_list[Y_A] = GRAVITY;                          /* m/s^2 */
		p_list_2[i].parm_list[Z_A] = 0;                                /* m/s^2 */

	}

	for(i=0; i < NUM_PARTICLES; i++) {
		calc_resting(&p_list_1[i],&precompute_path_list[i]);
		calc_resting(&p_list_2[i],&precompute_path_list[i+NUM_PARTICLES]);
	}

	sort_particles(precompute_path_list,NUM_PARTICLES*2,interacting_list,non_interacting_list);

	for(i=0; i < NUM_PARTICLES*2; i++) {
		printf("%i\n",interacting_list[i]);
	}
	printf("\n");
	for(i=0; i < NUM_PARTICLES*2; i++) {
		printf("%i\n",non_interacting_list[i]);
	}

	write_file(&png);
	close_file(&png);

	return(0);
}

void sort_particles(path2d* path_list, const int size_path_list, int* interacting, int* non_interacting) {
	int i,j;
	int interacting_hash[size_path_list];
	int non_interacting_hash[size_path_list];
	int next_interact_pos, next_non_interact_pos;

	if(path_list == NULL) { return; }
	if(interacting == NULL) { return; }
	if(non_interacting == NULL) { return; }

	memset(interacting_hash,0,sizeof(int)*size_path_list);
	memset(non_interacting_hash,0,sizeof(int)*size_path_list);

	for(i=0; i < size_path_list; i++) {
		for(j=0; j < size_path_list; j++) {
			if(path_list[i].id != path_list[j].id) {
				if(calc_intersect(path_list[i].x0,path_list[i].x1,path_list[j].x0,path_list[j].x1,
								  path_list[i].y0,path_list[i].y1,path_list[j].y0,path_list[j].y1) == 1) {
					interacting_hash[path_list[i].id] = 1;
					break;
				}
			}
		}
		if(j >= size_path_list) { non_interacting_hash[path_list[i].id] = 1; }
	}

	next_interact_pos = 0; next_non_interact_pos = 0;
	for(i=0; i < size_path_list; i++) {
		if(interacting_hash[i] == 1) {
			interacting[next_interact_pos++] = i;
		}
		if(non_interacting_hash[i] == 1) {
			non_interacting[next_non_interact_pos++] = i;
		}
	}

}

void calc_resting(particle* p, path2d* resting) {
	float t;

	if(p == NULL) { return; }
	if(resting == NULL) { return; }

	t = (p->parm_list[Y_V0] + sqrt(SQUARE(p->parm_list[Y_V0]) - (2.0F * p->parm_list[Y_A] * p->parm_list[Y0]))) / p->parm_list[Y_A];
	resting->x0 = p->x;
	resting->y0 = p->z;
	resting->x1 = (p->parm_list[X_V0] * t) + (p->parm_list[X_A] * 0.5F * t * t) + p->parm_list[X0];
	resting->y1 = (p->parm_list[Z_V0] * t) + (p->parm_list[Z_A] * 0.5F * t * t) + p->parm_list[Z0];
	resting->id = p->id;
}

BOOL calc_intersect(float Ax,float Bx,float Cx,float Dx,float Ay,float By,float Cy,float Dy) {
	if((By * Cx - Ay * Cx + Ax * Cy - By * Cy + Ay * Bx - Ax * By) * (By * Dx - Ay * Dx + Ax * Dy - By * Dy + Ay * Bx - Ax * By) > 0 &&
	   (Dy * Ax - Cy * Ax + Cx * Ay - Dy * Ay + Cy * Dx - Cx * Dy) * (Dy * Bx - Cy * Bx + Cx * By - Dy * By + Cy * Dx - Cx * Dy) > 0) {
		return(1);
	} else {
		return(0);
	}
}
