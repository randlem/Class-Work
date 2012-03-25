/************************************************
*
* Bank Teller Simulation with Queue Jumping
*
* This is a straight forward implementation of the teller simulation from the
* book.  Follows the pattern shown in the office demo.
*
* Written by: Mark Randles, Dan Sinclair
*
************************************************/

#include <iostream>
using std::cout;
using std::endl;

#include <vector>
using std::vector;

#include "SimPlus.h"

#include "Door.h"
#include "Teller.h"
#include "Customer.h"

const int    NUMB_TELLERS = 7;       // number of tellers in the simulation
const double CLOSING_TIME = 28800.0; // the time in seconds that the door will be "open" (8 hours)

int main() {
	SimPlus* simPlus = SimPlus::getInstance();  // an instance of the SimPlus lib
	vector<EntityQueue*> tellerQueue;
	vector<Teller*> tellers;
	int i;

	// create the teller queues and store their pointers for future use
	for(i=0; i < NUMB_TELLERS; ++i)
		tellerQueue.push_back(simPlus->getEntityQueue());

	// create the door or entry point for entites
	Door door(60.0,CLOSING_TIME,tellerQueue);

	// create all the tellers and save their pointers
	for(i=0; i < NUMB_TELLERS; ++i)
		tellers.push_back(new Teller(270.0,CLOSING_TIME,i,tellerQueue[i]));

	// loop until no more events are waiting in the queue
	do {
		if(simPlus->timing() != NULL)
			simPlus->reportError("unhandled event.");
	} while(simPlus->getEventListSize() > 0);

	// output the statistics for the tellers
	for(vector<Teller*>::iterator j=tellers.begin(); j != tellers.end(); ++j)
		cout << *(*j) << endl;

	// cleanup my memory
	for(vector<Teller*>::iterator j=tellers.begin(); j != tellers.end(); ++j)
		delete *j;

	// delete the instance of SimPlus
	delete simPlus;

	// return to the OS
	return(0);
}
