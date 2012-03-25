#include <stdio.h>
#include <mpi.h>

enum state { WALK, EAT, TALK };
typedef state;

typedef struct {
	int id;
	state curr_state;
	int x;
	int y;
	int time_since_eat;
} pik;

void pack_pik(pik* pik, char* buffer, int size_buffer) {
	int pos = 0;
	MPI_Pack(&pik->id,1,MPI_INT,buffer,size_buffer,&pos,MPI_COMM_WORLD);
	MPI_Pack(&pik->curr_state,1,MPI_INT,buffer,size_buffer,&pos,MPI_COMM_WORLD);
	MPI_Pack(&pik->x,1,MPI_INT,buffer,size_buffer,&pos,MPI_COMM_WORLD);
	MPI_Pack(&pik->y,1,MPI_INT,buffer,size_buffer,&pos,MPI_COMM_WORLD);
	MPI_Pack(&pik->time_since_eat,1,MPI_INT,buffer,size_buffer,&pos,MPI_COMM_WORLD);
}

void unpack_pik(pik* pik, char* buffer, int size_buffer) {
	int pos = 0;
	MPI_Unpack(buffer,size_buffer,&pos,&pik->id,1,MPI_INT,MPI_COMM_WORLD);
	MPI_Unpack(buffer,size_buffer,&pos,&pik->curr_state,1,MPI_INT,MPI_COMM_WORLD);
	MPI_Unpack(buffer,size_buffer,&pos,&pik->x,1,MPI_INT,MPI_COMM_WORLD);
	MPI_Unpack(buffer,size_buffer,&pos,&pik->y,1,MPI_INT,MPI_COMM_WORLD);
	MPI_Unpack(buffer,size_buffer,&pos,&pik->time_since_eat,1,MPI_INT,MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
	int mpi_id,           // this node's id
	    mpi_count;        // number of nodes
	pik da_pik;           // test pik
	char buffer[20];      // buffer
	MPI_Status status;

	// seed the random number generator
	srand(time(NULL));

	// init the MPI interface
	MPI_Init(&argc,&argv);

	// get some MPI stats
	MPI_Comm_rank(MPI_COMM_WORLD,&mpi_id);
	MPI_Comm_size(MPI_COMM_WORLD,&mpi_count);

	da_pik.id = 10;
	da_pik.curr_state = EAT;
	da_pik.x = 100;
	da_pik.y = 123;
	da_pik.time_since_eat = 3;

	if(mpi_id == 0) {
		pack_pik(&da_pik,buffer,20);
		MPI_Send(buffer,20,MPI_PACKED,1,0,MPI_COMM_WORLD);
	} else {
		MPI_Recv(buffer,20,MPI_PACKED,0,0,MPI_COMM_WORLD,&status);
		unpack_pik(&da_pik,buffer,20);
		printf("%i ",da_pik.id);
		if(da_pik.curr_state == WALK)
			printf("WALK\n");
		else if(da_pik.curr_state == TALK)
			printf("TALK\n");
		else
			printf("EAT\n");
	}

	// shutdown MPI
	MPI_Finalize();

}
