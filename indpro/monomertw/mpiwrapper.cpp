#include <vector>
using std::vector;

#include <mpi.h>
#include <stdlib.h>

#include "mpiwrapper.h"
#include "latprim.h"
#include "exception.h"
#include "event.h"

MPIWrapper::MPIWrapper() : rank(-1), nodeCount(-1), isInit(false), left(-1), right(-1), countSend(0), countRecv(0),countSendAnti(0),countRecvAnti(0) { ; }

MPIWrapper::~MPIWrapper() {
	//if(isInit)
	//	shutdown();
}

bool MPIWrapper::init(int* argv, char** argc[]) {
	MPI_Aint* displacements;
	MPI_Datatype* dataTypes;
	int* blockLength;
	MPI_Aint startAddress;
	MPI_Aint address;
	point p;
	site s;

	// see if init == true, if that is so we've got big problems
	if(isInit)
		throw(Exception("ERROR: Duplicate call to MPIWrapper::init()!"));

	// call MPI_Init() to start this whole shebang
	MPI_Init(argv,argc);

	// get the process rank and the number of nodes
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&nodeCount);

	// make sure the shit didn't hit the fan
	if(rank < 0) {
		throw(Exception("ERROR: MPI_Comm_rank() failed to return useful value!"));
	}
	if(nodeCount < 0) {
		throw(Exception("ERROR: MPI_Comm_size() filed to return useful value!"));
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

	MPI_Type_struct(2,blockLength,displacements,dataTypes,&typePoint);
	MPI_Type_commit(&typePoint);

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
	dataTypes[0] = typePoint;
	dataTypes[1] = MPI_INT;
	dataTypes[2] = MPI_INT;

	MPI_Address(&s.p,&startAddress);
	displacements[0] = 0;
	MPI_Address(&s.listIndex,&address);
	displacements[1] = address - startAddress;
	MPI_Address(&s.h,&address);
	displacements[2] = address - startAddress;

	MPI_Type_struct(3,blockLength,displacements,dataTypes,&typeSite);
	MPI_Type_commit(&typeSite);

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
	dataTypes[0] = typeSite;
	dataTypes[1] = typeSite;
	dataTypes[2] = MPI_DOUBLE;
	dataTypes[3] = MPI_INT;

	MPI_Address(&m.oldSite,&startAddress);
	displacements[0] = 0;
	MPI_Address(&m.newSite,&address);
	displacements[1] = address - startAddress;
	MPI_Address(&m.time,&address);
	displacements[2] = address - startAddress;
	MPI_Address(&m.type,&address);
	displacements[3] = address - startAddress;

	MPI_Type_struct(4,blockLength,displacements,dataTypes,&typeMessage);
	MPI_Type_commit(&typeMessage);

	delete [] displacements;
	delete [] dataTypes;
	delete [] blockLength;

	// attach the buffer to the MPI process
	MPI_Buffer_attach(malloc(BUFFER_SIZE_COUNT * sizeof(message) + MPI_BSEND_OVERHEAD), BUFFER_SIZE_COUNT * sizeof(message) + MPI_BSEND_OVERHEAD);

	// get the node on my left
	left = LEFT(rank,nodeCount);

	// get the node on my right
	right = RIGHT(rank,nodeCount);

	// hey, we finished the init!  so set the flag
	isInit = true;

	// return the value of the flag (should be true)
	return(isInit);
}

bool MPIWrapper::shutdown() {
	// make sure we had a successful init() call
	if(!isInit)
		return(false);

	// detach the buffer from the MPI process (COULD STALL PROGRAM EXECUTION
	// SINCE ALL BUFFERED MESSAGES MUST BE DELIVERED BEFORE THE CALL EXITS)
	MPI_Buffer_detach(&buffer,&bufferSize);

	// free the declared types
	MPI_Type_free(&typeMessage);
	MPI_Type_free(&typeSite);
	MPI_Type_free(&typePoint);

	// call the MPI_Finalize() function to make MPI clean up
	MPI_Finalize();

	// set init to false so we don't do anything stupid
	isInit = false;

	// return true so all is well
	return(!isInit);
}

bool MPIWrapper::sendMessage(message* m, Direction dir) {

	// send the message with a buffered send so we don't block
	if(DIR(dir) != -1) {
		MPI_Bsend(m, 1, typeMessage, DIR(dir), TAG_MESSAGE, MPI_COMM_WORLD);
		++countSend;
	}

	// return true
	return(true);
}

bool MPIWrapper::recvMessages(vector<message>* messages) {

	// loop until we don't have any more messages waiting
	while(isMessage()) {
		// recieve the message
		MPI_Recv(&m, 1, typeMessage, MPI_ANY_SOURCE, TAG_MESSAGE, MPI_COMM_WORLD, &status);
		messages->push_back(m);
		++countRecv;
	}

	// return true
	return(true);
}

bool MPIWrapper::isMessage() {
	// do an iprobe to get the value of flag (TRUE OR FALSE)
	MPI_Iprobe(MPI_ANY_SOURCE, TAG_MESSAGE, MPI_COMM_WORLD, &flag, &status);

	// return the value compared to the true equiv of 1 (since it's an int)
	return(flag == 1);
}

bool MPIWrapper::sendAntiMessage(message* m, Direction dir) {

	// send the message with a buffered send so we don't block
	if(DIR(dir) != -1) {
		MPI_Bsend(m, 1, typeMessage, DIR(dir), TAG_ANTI_MESSAGE, MPI_COMM_WORLD);
		++countSendAnti;
	}

	// return true
	return(true);
}

bool MPIWrapper::recvAntiMessages(vector<message>* messages) {

	// loop until we don't have any more messages waiting
	while(isAntiMessage()) {
		// recieve the message
		MPI_Recv(&m, 1, typeMessage, MPI_ANY_SOURCE, TAG_ANTI_MESSAGE, MPI_COMM_WORLD, &status);
		messages->push_back(m);
		++countRecvAnti;
	}

	// return true
	return(true);
}

bool MPIWrapper::isAntiMessage() {
	// do an iprobe to get the value of flag (TRUE OR FALSE)
	MPI_Iprobe(MPI_ANY_SOURCE, TAG_ANTI_MESSAGE, MPI_COMM_WORLD, &flag, &status);

	// return the value compared to the true equiv of 1 (since it's an int)
	return(flag == 1);
}

float MPIWrapper::allReduceFloat(float input, MPI_Op op) {
	float output;

	// call MPI_Allreduce() using the provided input/output, the correct datatype
	// and the user-provided op for the world communicator
	MPI_Allreduce(&input,&output,1,MPI_FLOAT,op,MPI_COMM_WORLD);

	// return the output value
	return(output);
}

double MPIWrapper::allReduceDouble(double input, MPI_Op op) {
	double output;

	// call MPI_Allreduce() using the provided input/output, the correct datatype
	// and the user-provided op for the world communicator
	MPI_Allreduce(&input,&output,1,MPI_DOUBLE,op,MPI_COMM_WORLD);

	// return the output value
	return(output);
}

int MPIWrapper::allReduceInt(int input, MPI_Op op) {
	int output;

	// call MPI_Allreduce() using the provided input/output, the correct datatype
	// and the user-provided op for the world communicator
	MPI_Allreduce(&input,&output,1,MPI_INT,op,MPI_COMM_WORLD);

	// return the output value
	return(output);
}

bool MPIWrapper::isRoot() {
	// return the value of this compare
	return(rank == ROOT_RANK);
}

void MPIWrapper::barrier() {
	MPI_Barrier(MPI_COMM_WORLD);
}

double MPIWrapper::wallTime() {
	return(MPI_Wtime());
}
