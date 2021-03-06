#include "lattice.h"
extern MPIWrapper mpi;
void sendmsgs(Lattice  * newlatt)
{

	mpi.sendboundaryevent(newlatt->bdyevent[left],newlatt->bdycount[left],newlatt->nbhr[left]);
	mpi.sendboundaryevent(newlatt->bdyevent[right],newlatt->bdycount[left],newlatt->nbhr[right]);

	mpi.recvboundaryevent(newlatt->bdyeventrec[left],newlatt->nbhr[left]);
	mpi.recvboundaryevent(newlatt->bdyeventrec[right],newlatt->nbhr[right]);

	for(int j=0;j<2;j++)
	{
		newlatt->bdycount[j]=0;
	}
}
