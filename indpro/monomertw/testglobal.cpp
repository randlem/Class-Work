#include <iostream>
using std::cout;
using std::endl;

#include "mpiwrapper.h"
#include "randgen.h"

int main(int argc, char* argv[]) {
	RandGen* rng;
	MPIWrapper mpi;
	double mGT,t,maxGT;
	int i=0;
	
	mpi.init(&argc,&argv);
	
	rng = new RandGen(1000,mpi.getRank());
	
	maxGT = mGT = t = 0.0;
	
	while(mGT < 100.0) {
		t += rng->getRandom(0.0);
		
		if(i > 10) {
			mpi.barrier();
			mGT = mpi.allReduceDouble(t,MPI_MIN);
			maxGT = mpi.allReduceDouble(t,MPI_MAX);
			if(mpi.isRoot())
				cout << mGT << " " << maxGT << endl;
			i=0;
		}
		++i;
	}

	mpi.shutdown();

}
