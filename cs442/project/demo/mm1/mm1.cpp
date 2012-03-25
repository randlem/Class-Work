#include <iostream>
using std::cout;
using std::endl;

#include "SimPlus.h"
#include "Server.h"
#include "Object.h"
#include "Entry.h"

int main() {
	SimPlus* simPlus = SimPlus::getInstance();
	EntityQueue* serverQueue = simPlus->getEntityQueue();
	double meanInterarrivalTime = 0.4;
	int numberOfEntities = 100000;
	double serverMeanServiceTime = 11.0;
	double serverSTSD = 2.0;

	EntryNode source(meanInterarrivalTime,numberOfEntities,serverQueue);
	Server server(serverMeanServiceTime,serverSTSD,serverQueue);

	while(server.getNumProcessed() < 1000) {
		//cout << server.getNumProcessed() << endl;
		if( simPlus->timing() != 0 )
			simPlus->reportError( "Unhandled event." );
	}

	Object::getStats(cout);
	cout << endl << server;

	// cleanup
	delete simPlus;

	return(0);
}
