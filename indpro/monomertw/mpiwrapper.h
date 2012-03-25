#include <vector>
using std::vector;

#include <fstream>
using std::fstream;
using std::endl;

#include <iostream>
using std::cout;

#include <mpi.h>

#include "latprim.h"
#include "latconst.h"
#include "exception.h"
#include "event.h"

#ifndef MPIWRAPPER_H
#define MPIWRAPPER_H

#define LEFT(a,b) (((a - 1) >= 0) ? (a - 1) : (b-1))
#define RIGHT(a,b) (((a + 1) < b) ? (a + 1) : 0)
#define DIR(a) ((a == LEFT) ? left : right)
const int BUFFER_SIZE_COUNT = 1024*1024*10; // 10MB buffer (overkill, baby)
const int ROOT_RANK = 0;
const int NUM_NEIGHBORS = 2;

const int TAG_MESSAGE = 0;
const int TAG_ANTI_MESSAGE = 1;

enum Direction {LEFT,RIGHT};

typedef struct {
	site oldSite;
	site newSite;
	double time;
	EventType type;
} message;

class MPIWrapper {
public:
	MPIWrapper();
	~MPIWrapper();

	bool init(int*, char***);
	bool shutdown();

	bool sendMessage(message* , Direction);
	bool recvMessages(vector<message>*);
	bool isMessage();

	bool sendAntiMessage(message* , Direction);
	bool recvAntiMessages(vector<message>*);
	bool isAntiMessage();

	float allReduceFloat(float, MPI_Op);
	double allReduceDouble(double, MPI_Op);
	int allReduceInt(int, MPI_Op);

	void barrier();
	double wallTime();

	bool isRoot();

	int getRank() {
		return(rank);
	}

	int getNodeCount() {
		return(nodeCount);
	}

	void printStats(fstream& file) {
		file << "--- MPI STATS ---" << endl;
		file <<	"rank = " << rank << endl;
		file << "nodeCount = " << nodeCount << endl;
		file << "left = " << left << " right = " << right << endl;
		file << "Send Messages = " << countSend << " Recieved Messages = " << countRecv << endl;
		file << "Send Anti-Messages = " << countSendAnti << " Recieved Anti-Messages = " << countRecvAnti << endl;
		file << "Total Send = " << (countSend + countSendAnti) << " Total Recv = " << (countRecv + countRecvAnti) << endl;
		file << "-----------------" << endl << endl;
		file.flush();
	}

private:
	int rank;
	int nodeCount;
	bool isInit;

	char* buffer;
	int bufferSize;

	MPI_Datatype typeSite;
	MPI_Datatype typePoint;
	MPI_Datatype typeMessage;

	int left;
	int right;
	message m;

	MPI_Status status;
	int flag;

	int countSend;
	int countRecv;
	int countSendAnti;
	int countRecvAnti;
};

#endif
