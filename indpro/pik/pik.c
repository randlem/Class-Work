#include <mpi.h>

#include "pik.h"

const int EAT_THRESHOLD     = 10;
const int TALK_THRESHOLD    = 4;
const int MOVE_TO_THRESHOLD = 25;
const int MOVE_MIN          = 1;
const int MOVE_MAX          = 3;

static int count_eat  = 0;
static int count_talk = 0;
static int count_walk = 0;

static int packed_pos = 0;
static int unpacked_pos = 0;

#define RAND_RANGE(a,b) ((a)+(rand()%((b)-(a)+1)))

void init_pik(pik* da_pik, int id, int x, int y) {
	da_pik->id = id;
	da_pik->x = x;
	da_pik->y = y;
	da_pik->curr_state = WALK;
	da_pik->time_since_eat = 0;
	count_walk++;
}

void update_pik(pik* da_pik, pik* nearest_pik, int dist) {

	// if the current state is talk the pik
	// isn't doing anything
	if(da_pik->curr_state == TALK) {
		//printf("PIK %i TALK\n",da_pik->id);
		return;
	}

	// if the pik state is eating then set it
	// to walk
	if(da_pik->curr_state == EAT) {
		//printf("PIK %i EAT\n",da_pik->id);
		count_walk++; count_eat--;
		da_pik->curr_state = WALK;
	}

	// see if the pik is going to eat this round
	if(RAND_RANGE(1,EAT_THRESHOLD) == 1) { // random chance
		da_pik->curr_state = EAT;
		count_eat++; count_walk--;
		da_pik->time_since_eat = 0;
		return;
	} else {
		da_pik->time_since_eat++;
	}

	if(da_pik->time_since_eat > EAT_THRESHOLD) {
		da_pik->curr_state = EAT;
		count_eat++; count_walk--;
		da_pik->time_since_eat = 0;
		return;
	}

	// see if the pik will begin to talk
	if(dist < TALK_THRESHOLD) {
		count_talk++; if(da_pik->curr_state == WALK) count_walk--; else count_eat--;
		da_pik->curr_state = TALK;
		return;
	}

	// otherwise, move the pik
	if(dist < MOVE_TO_THRESHOLD) {
		// move towards the nearest pik
		int mv_x = RAND_RANGE(MOVE_MIN,MOVE_MAX);
		int mv_y = RAND_RANGE(MOVE_MIN,MOVE_MAX);

		if(da_pik->x > nearest_pik->x) {
			mv_x *= -1;
		}
		if(da_pik->y > nearest_pik->y) {
			mv_y *= -1;
		}

		da_pik->x += mv_x;
		da_pik->y += mv_y;
	} else {
		// move in a random direction
		da_pik->x += RAND_RANGE(MOVE_MIN,MOVE_MAX);
		da_pik->y += RAND_RANGE(MOVE_MIN,MOVE_MAX);
	}

	//printf("PIK %i WALK\n",da_pik->id);
	return;
}

void print_stats() {
	printf("WALK = %i\nEAT = %i\nTALK = %i\n",count_walk,count_eat,count_talk);
}

void pack_pik(pik* pik, char* buffer, int size_buffer) {
	MPI_Pack(&pik->id,1,MPI_INT,buffer,size_buffer,&packed_pos,MPI_COMM_WORLD);
	MPI_Pack(&pik->curr_state,1,MPI_INT,buffer,size_buffer,&packed_pos,MPI_COMM_WORLD);
	MPI_Pack(&pik->x,1,MPI_INT,buffer,size_buffer,&packed_pos,MPI_COMM_WORLD);
	MPI_Pack(&pik->y,1,MPI_INT,buffer,size_buffer,&packed_pos,MPI_COMM_WORLD);
	MPI_Pack(&pik->time_since_eat,1,MPI_INT,buffer,size_buffer,&packed_pos,MPI_COMM_WORLD);
}

void unpack_pik(pik* pik, char* buffer, int size_buffer) {
	MPI_Unpack(buffer,size_buffer,&unpacked_pos,&pik->id,1,MPI_INT,MPI_COMM_WORLD);
	MPI_Unpack(buffer,size_buffer,&unpacked_pos,&pik->curr_state,1,MPI_INT,MPI_COMM_WORLD);
	MPI_Unpack(buffer,size_buffer,&unpacked_pos,&pik->x,1,MPI_INT,MPI_COMM_WORLD);
	MPI_Unpack(buffer,size_buffer,&unpacked_pos,&pik->y,1,MPI_INT,MPI_COMM_WORLD);
	MPI_Unpack(buffer,size_buffer,&unpacked_pos,&pik->time_since_eat,1,MPI_INT,MPI_COMM_WORLD);
}

void gather_stats(pik* piks,int count) {
	int i;

	count_walk = 0;
	count_talk = 0;
	count_eat = 0;

	for(i=0; i < count; i++) {
		switch(piks[i].curr_state) {
			case WALK: {
				count_walk++;
			} break;
			case EAT: {
				count_eat++;
			} break;
			case TALK: {
				count_talk++;
			} break;
		}
	}
}
