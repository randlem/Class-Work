#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <fstream>
using std::ofstream;

#include "exception.h"
#include "lattice.h"
#include "mpiwrapper.h"

const int globalSyncThreshold = 1000;

string makeFileName(string,string,int);

int main(int argc, char* argv[]) {
	Lattice lattice;
	string pngFilename = "";
	string logFilename = "";
	fstream logFile;
	double minGlobalTime = 0.0;
	double maxGlobalTime = 0.0;
	int globalTimeCounter = 0;
	double gConvergence = 0.0;
	int eventCount = 0;

	// setup the lattice mpi stuff
	lattice.mpi.init(&argc,&argv);

	try {
		pngFilename = makeFileName("height-node","png",lattice.mpi.getRank());
		logFilename = makeFileName("log","txt",lattice.mpi.getRank());

		logFile.open(logFilename.c_str(),fstream::out|fstream::trunc);

		if(!logFile) {
			string error = "Couldn't open log file " + logFilename;
			throw(Exception(error));
		}

		lattice.setMinGlobalTime(0.0);

		lattice.mpi.barrier();

		// MAIN LOOP
		while(gConvergence < 1.0) {
			// retrive any remote events
			lattice.negoitateEvents(logFile);

			// do the next event
			lattice.doNextEvent();

			// see if it's time for a global sync
			if(globalTimeCounter > globalSyncThreshold) {
				lattice.mpi.barrier();

				lattice.negoitateEvents(logFile);

				// allreduce to find the min time
				minGlobalTime = lattice.mpi.allReduceDouble(lattice.getLocalTime(),MPI_MIN);
				maxGlobalTime = lattice.mpi.allReduceDouble(lattice.getLocalTime(),MPI_MAX);
				eventCount = lattice.mpi.allReduceInt(lattice.getEventCount(),MPI_SUM);

				// set the global time in the lattice
				lattice.setMinGlobalTime(minGlobalTime);

				// clear the counter
				globalTimeCounter = 0;

				// calculate the global convergence
				gConvergence = (double)eventCount/(double)(lattice.mpi.getNodeCount() * SIZE);

				if(lattice.mpi.isRoot()) {
					cout << minGlobalTime << " " << maxGlobalTime << " " << gConvergence << endl;
					cout.flush();
				}
			} else
				++globalTimeCounter;
		}

		logFile << "exit main loop" << endl;
		logFile.flush();

		lattice.mpi.barrier();

		// rollback to minimum global time

		//lattice.printLatticeHeight(logFile);
		logFile << "gCovergence = " << gConvergence << endl;
		lattice.printStats(logFile);
		lattice.createHeightMap(pngFilename);
		lattice.mpi.printStats(logFile);

		lattice.cleanup(logFile);

		logFile.close();

		lattice.mpi.barrier();

	} catch(Exception err) {
		cerr << err.error << endl;
	}

	// close the mpi stuff
	lattice.mpi.shutdown();

	return(0);
}

string makeFileName(string prefix, string ext, int rank) {
	string output = prefix + ".";
	output += (char)('a' + rank);
	return(output + "." + ext);
}


// testing main loop
/* int main(int argc, char* argv[]) {
	Lattice lattice;
	fstream logFile0,logFile1,logFile2,logFile3,logFile4;
	double time;

	try {
		logFile0.open("log0.txt",fstream::out|fstream::trunc);
		logFile1.open("log1.txt",fstream::out|fstream::trunc);
		logFile2.open("log2.txt",fstream::out|fstream::trunc);
		logFile3.open("log3.txt",fstream::out|fstream::trunc);
		logFile4.open("log4.txt",fstream::out|fstream::trunc);

		if(!logFile0 || !logFile1 || !logFile2 || !logFile3 || !logFile4) {
			string error = "Couldn't open log file!";
			throw(Exception(error));
		}

		lattice.setMinGlobalTime(0.0);

		// MAIN LOOP
		for(int i=0; i < 1000; ++i) {
			lattice.doNextEvent(logFile4);
		}

		lattice.printLatticeHeight(logFile0);
		lattice.printMonomerList(logFile0);
		lattice.printStats(logFile0);

		time = lattice.getLocalTime();

		for(int i=0; i< 10000; ++i) {
			lattice.doNextEvent(logFile2);
		}

		lattice.printLatticeHeight(logFile2);
		lattice.printMonomerList(logFile2);
		lattice.printStats(logFile2);

		lattice.rollback(time);

		lattice.printLatticeHeight(logFile1);
		lattice.printMonomerList(logFile1);
		lattice.printStats(logFile1);

		for(int i=0; i< 10000; ++i) {
			lattice.doNextEvent(logFile3);
		}

		lattice.printLatticeHeight(logFile3);
		lattice.printMonomerList(logFile3);
		lattice.printStats(logFile3);

		logFile0.close();
		logFile1.close();
		logFile2.close();
		logFile3.close();
		logFile4.close();

	} catch(Exception err) {
		cerr << err.error << endl;
	}

	return(0);
}
*/
