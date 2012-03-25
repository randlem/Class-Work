/************************************************
*
* Door.cpp
*
* Implementation of the Door entry point. Derived from the EntryEntity class in
* the office demo
*
* Written by: Mark Randles, Dan Sinclair
*
************************************************/
#include "Door.h"

const unsigned short Door::ARRIVAL = 2;

Door::Door( double theta, double doorCloseTime, vector<EntityQueue*>& daTellerQueue ) : ServerEntity() {
	// get a handle to the kernel
	SimPlus* handle = SimPlus::getInstance();

	// mean interarrival time is an exponential distribution
	interarrivalTime = handle->getExponentialDist(theta);

	// make sure we know where to send new Customers
	tellerQueue = daTellerQueue;

	// set the door close time
	this->doorCloseTime = doorCloseTime;

	// get Door started generating Customers
	generateEvent(ARRIVAL, interarrivalTime->getRandom());
}

Door::~Door() {
	for(vector<Customer*>::iterator i=customerPool.begin(); i != customerPool.end(); ++i) {
		delete (*i);
	}
}

Event* Door::generateEvent( const unsigned short& eventType, const double& eventTime ) {
	// do a sanity check on the event type we received
	if( eventType != ARRIVAL )
		return NULL;
	// then just call base class method
//	cout << "ARRIVAL " << eventTime << endl;

	return ServerEntity::generateEvent( eventType, eventTime );
}

void Door::processEvent( Event* anEvent ) {
	// REMEMBER TO ACCOUNT FOR PREFIXING WHEN HANDLING EVENTS OR THE
	// KERNEL WILL YELL AT YOU!!!! SEE ServerEntity.h FOR MORE INFO!
	if( anEvent->getEventType() == (getPrefix() + ARRIVAL) ) {
		// get a handle to the kernel
		SimPlus* handle = SimPlus::getInstance();
		double simTime = handle->getSimTime();

		// create a new customer an push it into the pool
		Customer* temp = new Customer();
		customerPool.push_back(temp);

		// set up the customer
		temp->enterSystem(simTime);
		temp->beginWait(simTime);

		// find the shortest queue lenght and insert the new customer into it
		EntityQueue* shortestQueue = NULL;
		for(vector<EntityQueue*>::iterator i=tellerQueue.begin(); i != tellerQueue.end(); ++i) {
			if(shortestQueue != NULL && shortestQueue->getSize() > (*i)->getSize())
				shortestQueue = *i;
			else if(shortestQueue == NULL)
				shortestQueue = *i;
		}
		shortestQueue->addLast(temp);

		// if the simTime is past the door close time, we're going to exit before
		// scheduling another customer
		if(simTime > doorCloseTime) {
			cout << "Close time @ " << simTime << endl << endl;
			return;
		}

		// generate the next customer arrival
		generateEvent(ARRIVAL, simTime + interarrivalTime->getRandom());
	}
	else {
		reportError("Door");
	}

	SimPlus::getInstance()->releaseEvent(anEvent);
}
