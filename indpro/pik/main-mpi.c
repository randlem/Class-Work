#include <stdio.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#include "pik.h"

#define RAND_RANGE(a,b) ((a)+(rand()%((b)-(a)+1)))
#define SQUARE(a) (pow(a,2))

const int T        = 100;
const int WORLD_X  = 1024;
const int WORLD_Y  = 1024;

int NUM_PIKS = 0;

int packed_pos;
int unpacked_pos;

int main(int argc, char* argv[]) {
	int mpi_id,         // this node's id
	    mpi_count;      // number of nodes
	int t,i,j,k;
	pik* piks = NULL;	      // full list of piks
	int** closest_pik = NULL; // list of closes piks id & distance
	char* buffer = NULL;      // pack/unpack data buffer
	int size_buffer = 0;      // the size of the data buffer
	int* counts=NULL;         // the # of piks each node will process (root)
	int count;                // the # of piks to process (child)
	int start;                // the first pik that should be processed (child)
	int send_flag=1;          // a flag to send the child data buffer
	MPI_Status status;        // MPI status structure

	// seed the random number generator
	srand(time(NULL));

	// init the MPI interface
	MPI_Init(&argc,&argv);

	// get some MPI stats
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_id);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_count);

	// make sure there are enough cmd line parameters
	if(argc < 2) {
		fprintf(stderr,"Program takes one parameter!\n");
		return(1);
	}

	// set the num of piks from the cmd line
	NUM_PIKS = atoi(argv[1]);

	// allocate the closest pik array
	closest_pik = (int**)malloc(sizeof(int*) * NUM_PIKS);
	for(i=0; i < NUM_PIKS; i++) {
		closest_pik[i] = (int*)malloc(sizeof(int*) * 2);
	}

	// allocate the piks
	piks = (pik*)malloc(sizeof(pik) * NUM_PIKS);

	// allocate the pack/unpack data buffer
	size_buffer = sizeof(pik) * NUM_PIKS;
	buffer = (char*)malloc(size_buffer);

	// (root) allocate the counts array
	counts = (int*) malloc(sizeof(int) * (mpi_count-1));

	// create the counts array
	if(mpi_id == 0) {
		int all = (int)((float)NUM_PIKS / (float)mpi_count);
		int extra = NUM_PIKS % mpi_count;

		for(i=0; i < mpi_count-1; i++) {
			counts[i] = all;
			if(extra <= 0) {
				break;
			} else {
				counts[i]++;
				extra--;
			}
		}
	}

	// init the piks
	if(mpi_id == 0) {
		for(i=0; i < NUM_PIKS; i++) {
			init_pik(&piks[i],i,RAND_RANGE(0,WORLD_X),RAND_RANGE(0,WORLD_Y));
		}
	}

	printf("%i: Init done!\n", mpi_id);

	MPI_Barrier(MPI_COMM_WORLD);

	// main loop; process each pik for the total time T
	for(t=0; t < T; t++) {

		if(mpi_id == 0) {
			printf("T = %i\n",t);
		}

		// send the full list to each node
		if(mpi_id == 0) {
			for(i=0; i < NUM_PIKS; i++) {
				pack_pik(&piks[i],buffer,size_buffer);
			}
			MPI_Bcast(buffer,size_buffer,MPI_PACKED,0,MPI_COMM_WORLD);
		} else {
			MPI_Bcast(buffer,size_buffer,MPI_PACKED,0,MPI_COMM_WORLD);
			for(i=0; i < NUM_PIKS; i++) {
				unpack_pik(&piks[i],buffer,size_buffer);
			}
			printf("%i: Pik list recieved\n",mpi_id);
		}

		// send the count and position of the piks that each node should process
		if(mpi_id == 0) {
			int pos = 0;
			for(i=1; i < mpi_count; i++) {
				MPI_Send(&counts[i-1],1,MPI_INT,i,0,MPI_COMM_WORLD);
				MPI_Send(&pos,1,MPI_INT,i,0,MPI_COMM_WORLD);
				printf("%i: Sent count and start to %i\n",mpi_id,i);
				pos += counts[i-1];
			}
		} else {
			MPI_Recv(&count,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
			MPI_Recv(&start,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
			printf("%i: Recieved count and start %i %i\n",mpi_id,count,start);
		}

		// process each pik
		if(mpi_id != 0) {
			packed_pos = 0;
			for(j=start; j < start+count; j++) {
				// init the current entry
				closest_pik[j][0] = -1;
				closest_pik[j][1] = 0;

				// find the closest pik and save it's index in piks and the distance
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
				// update the current pik
				update_pik(&piks[j],&piks[closest_pik[j][0]],closest_pik[j][1]);
				printf("%i: Updated pik %i\n",mpi_id,piks[j].id);
				pack_pik(&piks[j],buffer,size_buffer);
				printf("%i: Packed pik %i\n",mpi_id,piks[j].id);
			}
			printf("%i: Done processing\n",mpi_id);
		}

		// wait for all the processes to finish
		MPI_Barrier(MPI_COMM_WORLD);

		if(mpi_id == 0) {
			int pos;
			for(i=1; i < mpi_count; i++) {
				MPI_Send(&send_flag,1,MPI_INT,i,0,MPI_COMM_WORLD);
				MPI_Recv(&buffer,size_buffer,MPI_PACKED,i,0,MPI_COMM_WORLD,&status);
				unpacked_pos = 0;
				for(j=pos; j < pos + counts[i-1]; j++) {
					unpack_pik(&piks[j],buffer,size_buffer);
				}
				pos += counts[i-1];
				printf("%i: Recieved results from %i\n",mpi_id,i);
			}
		} else {
			MPI_Recv(&send_flag,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
			MPI_Send(&buffer,size_buffer,MPI_PACKED,0,0,MPI_COMM_WORLD);
			printf("%i: Sent results\n",mpi_id);
		}

		// print the current counts for the world piks
		//gather_stats(piks,NUM_PIKS);
		//print_stats();
		printf("\n");
	}

	// shutdown MPI
	MPI_Finalize();

	return(0);
}
