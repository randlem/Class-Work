#include <iostream>
using std::cout;
using std::endl;

#include "mpiwrapper.h"
#include "boundryevent.h"

int main(int argc, char* argv[]) {
	MPIWrapper mpi;
	boundryEvent be;

	// init the mpi wrapper
	mpi.init(&argc,&argv);

	// test the send/recieve functionality


	// shutdown the mpi wrapper
	mpi.shutdown();

	// return UNIX true
	return(0);
}