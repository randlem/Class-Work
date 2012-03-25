#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include "png_writer.h"

#define PI 3.1415926F
#define TWO_PI 6.2831852F
#define HALF_PI 1.5707963F
#define NEG_HALF_PI -1.5707963F

#define RAND_RANGE(a,b) ((a)+(rand()%((b)-(a)+1)))
#define FP_RAND_RANGE(a,b) (a + ((b - a) * (random()/(float)LONG_MAX)))
#define DEG_TO_RAD(a) ((a * 3.1415926F) / 180)

typedef struct {
	float x;
	float y;
	float z;
} point;

typedef struct {
	point *vertex;
	int cnt;
} vertex_list;

int main(int argc, char* argv[]) {
	png_file png_xz,png_yz,png_xy;
	float diameter = 100.0F;
	float alpha,theta;
	float min_theta,max_theta,theta_inc;
	float min_y_dev,max_y_dev;
	int i;
	vertex_list v_list;

	if(argc != 3)
		return(1);

	srand(time(NULL));

	open_file(&png_xz,"xz.png",1024,1024,-150,150,-150,150);
	open_file(&png_yz,"yz.png",1024,1024,-150,150,-150,150);
	open_file(&png_xy,"xy.png",1024,1024,-150,150,-150,150);

	v_list.cnt = RAND_RANGE(atoi(argv[1]),atoi(argv[2]));
	v_list.vertex = (point*)malloc(sizeof(point)*v_list.cnt);
	for(i=0; i < v_list.cnt; i++) {
		alpha = FP_RAND_RANGE(0,PI);
		theta = FP_RAND_RANGE(0,TWO_PI);
		v_list.vertex[i].x = diameter * cos(alpha) * sin(theta);
		v_list.vertex[i].y = diameter * sin(alpha) * sin(theta);
		v_list.vertex[i].z = diameter * cos(theta);
	}

	for(i=0; i < v_list.cnt; i++) {
		plot_line(&png_xz,v_list.vertex[i].x,v_list.vertex[i].z,0,0,255,255,255);
		plot_line(&png_yz,v_list.vertex[i].y,v_list.vertex[i].z,0,0,255,255,255);
		plot_line(&png_xy,v_list.vertex[i].x,v_list.vertex[i].y,0,0,255,255,255);
	}

	write_file(&png_xz);
	write_file(&png_yz);
	write_file(&png_xy);

	close_file(&png_xz);
	close_file(&png_yz);
	close_file(&png_xy);

	return(0);
}
