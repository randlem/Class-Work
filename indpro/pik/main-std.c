#include <stdio.h>
#include <time.h>
#include <math.h>

#include "pik.h"

#define RAND_RANGE(a,b) ((a)+(rand()%((b)-(a)+1)))
#define SQUARE(a) (pow(a,2))

const int T        = 100;
const int WORLD_X  = 1024;
const int WORLD_Y  = 1024;

int NUM_PIKS = 0;

int main(int argc, char* argv[]) {
	int i,j,k;
	pik* piks = NULL;	      // full list of piks
	int** closest_pik = NULL; // list of closes piks id & distance

	// seed the random number generator
	srand(time(NULL));

	// set the num of piks from the cmd line
	if(argc < 2) {
		fprintf(stderr,"Program takes one parameter!\n");
		return(1);
	}

	NUM_PIKS = atoi(argv[1]);

	// allocate the closest pik array
	closest_pik = (int**)malloc(sizeof(int*) * NUM_PIKS);
	for(i=0; i < NUM_PIKS; i++) {
		closest_pik[i] = (int*)malloc(sizeof(int*) * 2);
	}

	// allocate the piks
	piks = (pik*)malloc(sizeof(pik) * NUM_PIKS);

	// init the piks
	for(i=0; i < NUM_PIKS; i++) {
		init_pik(&piks[i],i,RAND_RANGE(0,WORLD_X),RAND_RANGE(0,WORLD_Y));
	}

	// main loop; process each pik for the total time T
	for(i=0; i < T; i++) {
		printf("T = %i\n",i);
		for(j=0; j < NUM_PIKS; j++) {
			closest_pik[j][0] = -1;
			closest_pik[j][1] = 0;
			for(k=0; k < NUM_PIKS; k++) {
				int dist = (int)sqrt(SQUARE(piks[j].x - piks[k].x) + SQUARE(piks[j].y - piks[k].y));
				if(dist > 0) {
					if(closest_pik[j][1] > dist) {
						closest_pik[j][0] = k;
						closest_pik[j][1] = dist;
					} else if(closest_pik[j][1] == 0) {
						closest_pik[j][0] = k;
						closest_pik[j][1] = dist;
					}
				}
			}
			//printf("%i %i %i\n",j,closest_pik[j][0],closest_pik[j][1]);
			update_pik(&piks[j],&piks[closest_pik[j][0]],closest_pik[j][1]);
		}
		print_stats();
		printf("\n");
	}

	return(0);
}
