/* Mark Randles
*  CS417
*  Lab Assignment #1
*  Due 2004.02.18
*   
*  The purpose of this assignment was to demonstrate basic MPI skillz by 
*  multiplying two very large matricies togather in parallel.  Also to 
*  experement with program speedup vs number of nodes.
*/
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <mpi.h>

#define MATRIX_A_ROW 4000
#define MATRIX_A_COL 2000
#define MATRIX_B_ROW 2000
#define MATRIX_B_COL 5000
#define MATRIX_C_ROW 4000
#define MATRIX_C_COL 5000

int main(int argc, char* argv[]) {
	/* variables and stuff */
	unsigned int A[MATRIX_A_ROW][MATRIX_A_COL],
	    B[MATRIX_B_ROW][MATRIX_B_COL],
	    C[MATRIX_C_ROW][MATRIX_C_COL],
	    temp_C[MATRIX_C_ROW*2][MATRIX_C_COL];
	int numb_nodes,id;
	int partition_a,partition_root;
	int send_flg,recv_flg;
	int i,j,k,l;
	MPI_Status status;
	double t1,init_time,comm_1_time,comm_2_time,approx_run_time;

	/* MPI init stuff */
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&id);
	MPI_Comm_size(MPI_COMM_WORLD,&numb_nodes);

	/* init the matricies and do everything we can precompute 
	   here since it's unsafe to do it earlier */
	partition_root = MATRIX_A_ROW%(numb_nodes-1);
	partition_a = (MATRIX_A_ROW-partition_root)/(numb_nodes-1);
	memset(A,0,sizeof(int)*MATRIX_A_ROW*MATRIX_A_COL);
	memset(B,0,sizeof(int)*MATRIX_B_ROW*MATRIX_B_COL);
	memset(C,0,sizeof(int)*MATRIX_C_ROW*MATRIX_C_COL);
	memset(temp_C,0,sizeof(int)*MATRIX_C_ROW*2*MATRIX_C_COL);
	if(id == 0) {
		t1 = MPI_Wtime();
		printf("Part. Root=%i\nPart. Worker=%i\nTotal=%i\n",partition_root,
                                       partition_a,((numb_nodes-1)*partition_a)+partition_root);
		for(i=0; i < MATRIX_A_ROW; i++)
			for(j=0; j < MATRIX_A_COL; j++)
				A[i][j] = i+j+1;
		for(i=0; i < MATRIX_B_ROW; i++)
			for(j=0; j < MATRIX_B_COL; j++)
				B[i][j] = i+j+1;
		init_time = MPI_Wtime() - t1;
	}

	/* broadcast matrix B to all the nodes */
	MPI_Bcast(B,MATRIX_B_COL*MATRIX_B_ROW,MPI_INT,0,MPI_COMM_WORLD);

	if(id == 0) {
		/* send each piece of data to the workers */
		t1 = MPI_Wtime();
		for(i=1; i < numb_nodes; i++) {
			MPI_Send(A[(i-1)*partition_a],partition_a*MATRIX_A_COL,
						MPI_INT,i,0,MPI_COMM_WORLD);
			MPI_Send(B,MATRIX_B_COL*MATRIX_B_ROW,MPI_INT,i,0,MPI_COMM_WORLD);
		}
		comm_1_time = MPI_Wtime() - t1;

		/* since it's possible for us to end up with extra lines i'm going to let 
		   the root node take care of them because we'll always have in the worst case
		   partition_a-1 extra lines to calculate */
		t1 = MPI_Wtime();
		if(partition_root != 0) {
			for(i=(numb_nodes-1)*partition_a; i < ((numb_nodes-1)*partition_a)+partition_root; i++) {
				for(j=0; j < MATRIX_B_COL; j++) {
					int sum = 0;
					for(k=0; k < MATRIX_A_COL; k++)
						sum += A[i][k] * B[k][j];
					C[i][j] = sum;
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		approx_run_time = MPI_Wtime() - t1;


		/* it's not the most elegant solution but it works...ordered send and recieve. Each child
		   will block till the parent wants it to send the childs data. The parent will block till
		   the child sends the data then proceede to the next child. */
		t1 = MPI_Wtime();
		recv_flg = 1;
		for(i=1; i < numb_nodes; i++) {
			MPI_Send(&recv_flg,1,MPI_INT,i,0,MPI_COMM_WORLD);
			MPI_Recv(C[(i-1)*partition_a],partition_a*MATRIX_B_COL,MPI_INT,
			 	 i,0,MPI_COMM_WORLD,&status);
	
		}
		comm_2_time = MPI_Wtime() - t1;

	} else {
		/* wait for the data */
		MPI_Recv(A,partition_a*MATRIX_A_COL,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		MPI_Recv(B,MATRIX_B_ROW*MATRIX_B_COL,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		
		/* process the data */
		for(i=0; i < partition_a; i++) {
			for(j=0; j < MATRIX_B_COL; j++) {
				int sum = 0;
				for(k=0; k < MATRIX_A_COL; k++)
					sum += A[i][k] * B[k][j];
				C[i][j] = sum;
			}
		}

		/* wait for all the process to be done (yea i know this 
		   isn't needed, but it'll help with the timing) */
		MPI_Barrier(MPI_COMM_WORLD);
		
		/* block till the parent wants us to send...then send the data */
		MPI_Recv(&send_flg,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		MPI_Send(C,partition_a*MATRIX_B_COL,MPI_INT,0,0,MPI_COMM_WORLD);

	}	

	/* print out the total time to process...
	   i had a hellva time getting these to output right values...stupid printf... */
	if(id == 0) {
		printf("%i %i\n",C[0][0],C[MATRIX_C_ROW-1][0]);
		printf("Initlization time = %f\n",(float)init_time);
		printf("Data distro time  = %f\n",(float)comm_1_time);
		printf("Processing time   = %f\n",(float)approx_run_time);
		printf("Data gather time  = %f\n",(float)comm_2_time);
		printf("------------------------------\n");
		printf("TOTAL             = %f\n",(float)(init_time+comm_1_time+approx_run_time+comm_2_time));
	}

	/* shutdown MPI */
	MPI_Finalize();

	/* return to the system */
	return(0);
}
