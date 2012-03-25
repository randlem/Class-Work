#include <vector>
using std::vector;

#include <queue>
using std::priority_queue;

#include <stack>
using std::stack;

#include <iomanip>
using std::setw;
using std::hex;
using std::dec;
using std::setprecision;

#include <fstream>
using std::fstream;

#include <string>
using std::string;

#include <png.h>
#include <math.h>

#include "exception.h"
#include "latprim.h"
#include "latconst.h"
#include "event.h"
#include "randgen.h"
#include "rewindlist.h"
#include "mpiwrapper.h"

#ifndef LATTICE_H
#define LATTICE_H

#define GET_DIR(a) ((a < LEFT_X_BOUNDRY) ? LEFT : RIGHT)

class Lattice {
public:
	Lattice();
	~Lattice();

	void cleanup(fstream&);

	bool doNextEvent();

	double getLocalTime() {
		return(localTime);
	}

	bool setMinGlobalTime(double mGT) {
		minGlobalTime = mGT;
		return(true);
	}

	double getMinGlobalTime() {
		return(minGlobalTime);
	}

	bool negoitateEvents(fstream&);

	// DEBUG FUNCTIONS
	void printLatticeHeight(fstream& file) {
		for(int i=0; i < DIM_X + GHOST + GHOST; ++i) {
			for(int j=0; j < DIM_Y; ++j) {
				file << lattice[i][j].h << " ";
			}
			file << endl;
		}
		file << "------------------------------------------------" << endl << endl;
		file.flush();
	}

	void printLatticeIndex(fstream& file) {
		for(int i=0; i < DIM_X + GHOST + GHOST; ++i) {
			for(int j=0; j < DIM_Y; ++j) {
				if(lattice[i][j].listIndex >= 0)
					file << setw(4) << lattice[i][j].listIndex;
				else
					file << setw(4) << "x";
				file << " ";
			}
			file << endl;
		}
		file << "------------------------------------------------" << endl << endl;
		file.flush();
	}

	void printMonomerList(fstream& file) {
		file << "monomerList[" << monomerList.size() << "] at time=" << localTime << endl;
		for(int i=0; i < monomerList.size(); ++i) {
			site* s = monomerList[i];
			file << i << ": (" << s->p.x << "," << s->p.y << ") h=" << s->h << " listIndex=" << s->listIndex << " " << hex << s << dec << endl;
		}
		file << "------------------------------------------------" << endl << endl;
		file.flush();
	}

	void printStats(fstream& file) {
		file << setprecision(10) << endl;
		file << "COLLECTED STATISTICS" << endl;
		file << "----------------------" << endl;
		file << "Convergence = " << getConvergence() << endl;
		file << "Total Event Count = " << countEvents << endl;
		file << "Total Diffusion Events = " << countDiffusion << endl;
		file << "Total Deposition Events = " << (countEvents - countDiffusion) << endl;
		file << "Total Boundry Events = " << countBoundry << endl;
		file << "Total Number Remote Events = " << countRemote << endl;
		file << "Total Rollbacks Performed = " << countRollback << endl;
		file << "Monomer List Size = " << monomerList.size() << endl;
		file << "Local Time = " << localTime << endl;
		file << "Min Global Time = " << minGlobalTime << endl;
		file << "Size = " << SIZE << " DIM_X = " << DIM_X << " DIM_Y = " << DIM_Y << endl;
		file << "Next random = " << rng.peekRandom() << endl;
		file << "RNG used = " << rng.getIndex() << endl;
		file << "------------------------------------------------" << endl << endl;
		file.flush();
	}

	int getEventCount() { return(countEvents - countDiffusion); }

	bool createHeightMap(string filename);

	MPIWrapper mpi;  // I can't think of a better place for this.

	bool rollback(const double);

	double getConvergence() {
		return((double)(countEvents - countDiffusion) / (double)(DIM_X * DIM_Y));
	}

private:
	double computeTime();
	bool deposit();
	bool diffuse();
	bool doKMC();
	EventType getNextEventType();
	bool commitEvent(Event*);
	site* randomMove(site*);
	bool isBoundry(point);
	bool isBound(site*);
	bool clearBonded(site*,const double);
	bool translateMessages(vector<Event*>* , vector<message>*);
	message* makeMessage(Event*);
	bool hasAntiEvent(Event*);

	double localTime;    // the time local to the lattice
	double minGlobalTime; // the minimum Global time (point of no return)

	RewindList<site *> monomerList; // list of all unbound monomers
	//site lattice[DIM_X + GHOST + GHOST][DIM_Y];  // the lattice (the extra two are the ghost region)
	site** lattice;

	float depositionRate; // the deposition rate of monomers
	float diffusionRate;  // the diffusion rate of monomers

	int countDiffusion;
	int countEvents;
	int countBoundry;
	int countRemote;
	int countRollback;

	priority_queue<Event*> remoteEventList; // list of all the remote dep/diffusion events
	stack<Event*> eventList;                // stack of all events to rollback the simulation
	vector<Event*> antiEvents;              // list of anti-events that will occur in the future

	RandGen rng; // random number generator

	point movementDir[NUM_DIR]; // array of movement types
	message m; // message for sending events
};

#endif
