#include "Server.h"

Server::Server( double mu, double sigma, EntityQueue* myQueue) : ServerEntity() {
	SimPlus* handle = SimPlus::getInstance();
	idleTimeStat = handle->getSampST();
	waitTimeStat = handle->getSampST();
	serviceTimeStat = handle->getSampST();

	numProcessed = 0;

	myServiceTime = handle->getNormalDist( mu, sigma );

	bind(myQueue, "myQueue");

	generateEvent(BEGIN_SERVICE, getCheckDelay());
}

Server::~Server() {

}

Event* Server::generateEvent( const unsigned short& eventType,	const double& eventTime) {
	if( eventType != BEGIN_SERVICE && eventType != END_SERVICE )
		return 0;
	return ServerEntity::generateEvent( eventType, eventTime );
}

void Server::processEvent( Event* anEvent ) {
	double simTime = SimPlus::getInstance()->getSimTime();

	if( anEvent->getEventType() == (getPrefix() + BEGIN_SERVICE) ) {
		// see if there are any patients waiting
		currentObject = (Object*)getBoundQueue("myQueue")->removeFirst();

		// if no patients are waiting, check again in 1 minute
		if( currentObject == 0 ) {
			generateEvent( BEGIN_SERVICE, simTime + getCheckDelay() );
		}
		// otherwise go to work
		else {
			generateEvent( END_SERVICE, simTime +
				myServiceTime->getRandom() );
			lastStart = simTime;
			idleTimeStat->observe(simTime, simTime - lastStop );
		}

	}
	else if( anEvent->getEventType() == (getPrefix() + END_SERVICE) ) {
		currentObject = 0;

		lastStop = simTime;
		serviceTimeStat->observe(simTime, simTime - lastStart );

		//check for patients again in 1 minute
		generateEvent( BEGIN_SERVICE, simTime + getCheckDelay() );

		numProcessed++;
	}
	else {
		reportError( "Server" );
	}
	SimPlus::getInstance()->releaseEvent( anEvent );
}

ostream& operator<<( ostream& out, Server& theServer )
{
	out << "Server: Average idle time between patients: "
	    << theServer.idleTimeStat->getMean() << endl;
	out << "Server: Average service time for patients:  "
	    << theServer.serviceTimeStat->getMean() << endl;
	out << "Server: Average patient wait time in queue: "
	    << theServer.waitTimeStat->getMean() << endl << endl;
	return out;
}
