#include <mpi.h>

#include "mpiwrapper.h"
#include "boundryevent.h"
#include "site.h"
#include "point.h"

MPIWrapper::MPIWrapper() {
	// set the initial values
	rank = -1;
	nodeCount = -1;
	init = false;
}

MPIWrapper::~MPIWrapper() {
	if(init)
		shutdown();
}

bool MPIWrapper::init(int* argv, char** argc[]) {
	MPI_Aint* displacements;
	MPI_Datatype* dataTypes;
	int* blockLength;
	MPI_Aint* startAddress;
	MPI_Aint* address;
	point p;
	site s;
	boundryEvent be;

	// see if init == true, if that is so we've got big problems
	if(init)
		throw("ERROR: Duplicate call to MPIWrapper::init()!");

	// call MPI_Init() to start this whole shebang
	MPI_Init(argv,argc);

	// get the process rank and the number of nodes
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&nodeCount);

	// make sure the shit didn't hit the fan
	if(rank < 0) {
		throw("ERROR: MPI_Comm_rank() failed to return useful value!");
	}
	if(nodeCount < 0) {
		throw("ERROR: MPI_Comm_size() filed to return useful value!");
	}

	// create the datatype for the point structure
	displacements = new MPI_Aint[2];
	dataTypes = new MPI_Datatype[2];
	blockLength = new int[2];

	blockLength[0] = 1;
	blockLength[1] = 1;
	dataTypes[0] = MPI_INT;
	dataTypes[1] = MPI_INT;

	MPI_Address(&p.x,&startAddress);
	displacements[0] = 0;
	MPI_Address(&p.y,&address);
	displacements[1] = address - startAddress;

	MPI_Type_struct(2,blockLength,displacements,dataTypes,&pointType);
	MPI_Type_commit(&pointType);

	delete [] displacements;
	delete [] dataTypes;
	delete [] blockLength;

	// create the datatype for the site structure
	displacements = new MPI_Aint[3];
	dataTypes = new MPI_Datatype[3];
	blockLength = new int[3];

	blockLength[0] = 1;
	blockLength[1] = 1;
	blockLength[2] = 1;
	dataTypes[0] = pointType;
	dataTypes[1] = MPI_INT;
	dataTypes[2] = MPI_INT;

	MPI_Address(&s.position,&startAddress);
	displacements[0] = 0;
	MPI_Address(&s.index,&address);
	displacements[1] = address - startAddress;
	MPI_Address(&s.h,&address);
	displacements[2] = address - startAddress;

	MPI_Type_struct(3,blockLength,displacements,dataTypes,&siteType);
	MPI_Type_commit(&siteType);

	delete [] displacements;
	delete [] dataTypes;
	delete [] blockLength;

	// create the datatype for the boundryEvent structure
	displacements = new MPI_Aint[4];
	dataTypes = new MPI_Datatype[4];
	blockLength = new int[4];

	blockLength[0] = 1;
	blockLength[1] = 1;
	blockLength[2] = 1;
	blockLength[3] = 1;
	dataTypes[0] = siteType;
	dataTypes[1] = siteType;
	dataTypes[2] = MPI_DOUBLE;
	dataTypes[3] = MPI_INT;

	MPI_Address(&be.oldSite,&startAddress);
	displacements[0] = 0;
	MPI_Address(&be.newSite,&address);
	displacements[1] = address - startAddress;
	MPI_Address(&be.time,&address);
	displacements[2] = address - startAddress;
	MPI_Address(&be.tag,&address);
	displacements[3] = address - startAddress;

	MPI_Type_struct(4,blockLength,displacements,dataTypes,&boundryEventType);
	MPI_Type_commit(&boundryEventType);

	delete [] displacements;
	delete [] dataTypes;
	delete [] blockLength;

	// attach the buffer to the MPI process
	MPI_Attach_buffer(new (bufferSizeCount * sizeof(boundryEvent) + MPI_BSEND_OVERHEAD), bufferSizeCount * sizeof(boundryEvent) + MPI_BSEND_OVERHEAD);

	// hey, we finished the init!  so set the flag
	init = true;

	// return the value of the flag (should be true)
	return(init);
}

bool MPIWrapper::shutdown() {
	// make sure we had a successful init() call
	if(!init)
		return(false);

	// detach the buffer from the MPI process (COULD STALL PROGRAM EXECUTION
	// SINCE ALL BUFFERED MESSAGES MUST BE DELIVERED BEFORE THE CALL EXITS)
	MPI_Detach_buffer(&buffer,&bufferSize);

	// free the declared types
	MPI_Type_free(&boundryEventType);
	MPI_Type_free(&siteType);
	MPI_Type_free(&pointType);

	// call the MPI_Finalize() function to make MPI clean up
	MPI_Finalize();

	// set init to false so we don't do anything stupid
	init = false

	// return true so all is well
	return(true);
}

bool MPIWrapper::sendBoundryEvent(boundryEvent* be, int destination) {
	// do a buffered send call
	MPI_Bsend(be, 1, boundryEventType, destination, 0, MPI_COMM_WORLD);

	// return true
	return(true);
}

bool MPIWrapper::recvBoundryEvent(boundryEvent* be, int source) {
	// recieve a message
	MPI_Recv(be, 1, boundryEventType, source, 0, MPI_COMM_WORLD, &status);

	// return true
	return(true);
}

bool isRoot() {
	// return the value of this compare
	return(rank == rootRank);
}
