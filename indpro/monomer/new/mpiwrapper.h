#include <mpi.h>

#include "boundryevent.h"

#ifndef MPIWRAPPER_H
#define MPIWRAPPER_H

const int bufferSizeCount = 1024;
const int rootRank = 0;

class MPIWrapper {
public:
	MPIWrapper();
	~MPIWrapper();

	bool init(int*, char**);
	bool shutdown();

	bool sendBoundryEvent(boundryEvent*, int);
	bool recvBoundryEvent(boundryEvent*, int);

	bool isRoot();

private:
	int rank;
	int nodeCount;
	bool init;

	char* buffer;
	int bufferSize;

	MPI_Datatype boundryEventType;
	MPI_Datatype siteType;
	MPI_Datatype pointType;
	MPI_Status status;
};

#endif