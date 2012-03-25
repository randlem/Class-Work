#include <stdio.h>
#include <mpi.h>

int main(int argv,char* argc[]) {
	int id=0,numb_proc=0;

	MPI_Init(&argv,&argc);

	MPI_Comm_rank(MPI_COMM_WORLD,&id);
	MPI_Comm_size(MPI_COMM_WORLD,&numb_proc);

	printf("%i %i %i",id,numb_proc,argv);

	MPI_Finalize();

	return(0);
}
