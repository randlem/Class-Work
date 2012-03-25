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

void break_vertex(vertex_list vertex);

int main(int argc, char* argv[]) {
	png_file png;
	float diameter = 100.0F;
	float alpha,theta;
	float min_theta,max_theta,theta_inc;
	float min_y_dev,max_y_dev;
	int i;
	vertex_list* old_list = NULL;
	vertex_list* new_list = NULL;

	if(argc != 3)
		return(1);

	srand(time(NULL));

	open_file(&png,"test.png",1024,1024,-150,150,-150,150);

	/* prime the inital halves by generating random points along a median */
	if(old_list != NULL)
		free(old_list);
	old_list = (vertex_list*)malloc(sizeof(vertex_list)*2);
	old_list[0].cnt = RAND_RANGE(atoi(argv[1]),atoi(argv[2]));
	old_list[0].vertex = (point*)malloc(sizeof(point)*old_list[0].cnt);
	min_theta=0; max_theta = theta_inc = TWO_PI/(float)old_list[0].cnt;
	min_y_dev = -.087266;  max_y_dev= .087266;
	for(i=0; i < old_list[0].cnt; i++) {
		alpha = FP_RAND_RANGE(0,max_y_dev);
		theta = FP_RAND_RANGE(min_theta,max_theta);
		old_list[0].vertex[i].x = diameter * cos(alpha) * sin(theta);
		old_list[0].vertex[i].y = diameter * sin(alpha) * sin(theta);
		old_list[0].vertex[i].z = diameter * cos(theta);
		min_theta+=theta_inc; max_theta+=theta_inc;
	}
	old_list[1].cnt = old_list[0].cnt;
	old_list[1].vertex = (point*)malloc(sizeof(point)*old_list[1].cnt);
	memcpy(old_list[1].vertex,old_list[0].vertex,sizeof(point)*old_list[0].cnt);

	for(i=0; i < old_list[0].cnt; i++) {
		if(i == 0)
			plot_line(&png,old_list[0].vertex[i].x,old_list[0].vertex[i].z,
					  old_list[0].vertex[old_list[0].cnt-1].x,old_list[0].vertex[old_list[0].cnt-1].z,255,255,255);
		else
			plot_line(&png,old_list[0].vertex[i].x,old_list[0].vertex[i].z,
		              old_list[0].vertex[i-1].x,old_list[0].vertex[i-1].z,255,255,255);
	}

	write_file(&png);

	close_file(&png);

	return(0);
}

void break_vertex(vertex_list* old_piece, vertex_list* new_pieces, int direction, int break_pts) {
	vertex_list break_line;
	float min_alpha,max_alpha,alpha_inc;

	break_line.vertex = (point*)malloc(sizeof(point)*break_pts);
	break_line.cnt = break_pts;
	min_alpha
	for(i=0; i < break_pts; i++) {
		alpha = FP_RAND_RANGE();
		theta = FP_RAND_RANGE(0,DEG_TO_RAD(2.0F));
		old_list[0].vertex[i].x = diameter * cos(alpha) * sin(theta);
		old_list[0].vertex[i].y = diameter * sin(alpha) * sin(theta);
		old_list[0].vertex[i].z = diameter * cos(theta);
		min_alpha+=alpha_inc; max_alpha+=alpha_inc;
	}

}
