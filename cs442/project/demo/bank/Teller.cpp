/************************************************
*
* Teller.cpp
*
* Implementation of the Teller ServerEntity.  Also handles the queue jumping functionality.
*
* Written by: Mark Randles, Dan Sinclair
*
************************************************/
#include "Teller.h"

vector<Teller*> Teller::tellerList;

Teller::Teller(double mu, double cT, int id, EntityQueue* doorQueue) : ServerEntity() {
	SimPlus* handle = SimPlus::getInstance();
	idleTimeStat = handle->getSampST();
	waitTimeStat = handle->getSampST();
	serviceTimeStat = handle->getSampST();
	closingTime = cT;
	tellerID = id;
	customerCount = 0;

	myServiceTime = handle->getExponentialDist( mu );

	lastStart = 0;
	lastStop = 0;

	bind(doorQueue, "MY_QUEUE");

	tellerList.push_back(this);

	generateEvent( BEGIN_SERVICE, getCheckDelay() );
}

Teller::~Teller() {

}

Event* Teller::generateEvent(const unsigned short& eventType, const double& eventTime) {
	if(eventType != BEGIN_SERVICE && eventType != END_SERVICE)
		return(NULL);
	return ServerEntity::generateEvent( eventType, eventTime );
}

void Teller::processEvent(Event* anEvent) {
	double simTime = SimPlus::getInstance()->getSimTime();

	if(anEvent->getEventType() == (getPrefix() + BEGIN_SERVICE)) {
		// see if there are any patients waiting
		currentCustomer = (Customer*)getBoundQueue("MY_QUEUE")->removeFirst();

		if(currentCustomer == NULL) {
			// if no customers are waiting, check again in 1 minute
			if(simTime < closingTime)
				generateEvent( BEGIN_SERVICE, simTime + getCheckDelay() );
		} else { // otherwise go to work
			generateEvent(END_SERVICE, simTime + myServiceTime->getRandom());
			lastStart = simTime;
			idleTimeStat->observe(simTime, simTime - lastStop);
		}

	} else if(anEvent->getEventType() == (getPrefix() + END_SERVICE))	{
		// increate the a coustomer processed counter
		++customerCount;

		currentCustomer = NULL;

		// collect some stats
		lastStop = simTime;
		serviceTimeStat->observe(simTime, simTime - lastStart);

		// generate a new service event
		generateEvent(BEGIN_SERVICE, simTime + getCheckDelay());

		// see if a customer jockies
		jockey();

	} else {
		reportError( "Teller" );
	}

	SimPlus::getInstance()->releaseEvent( anEvent );

}

int Teller::getQueueSize() {
	return(getBoundQueue("MY_QUEUE")->getSize());
}

void Teller::jockey() {
	vector<Teller*>::iterator i;
	Teller* jockeyFrom = NULL;
	int ni = getQueueSize();
	int nj = 0;
	unsigned int min_distance = 0xFFFFFFFF;
	unsigned int distance;

	// caculate the distance between the customer and jocky, recording if there is a jocky avaliabe
	for(i=tellerList.begin(); i != tellerList.end(); ++i) {
		nj = (*i)->getQueueSize();
		distance = abs((*i)->getId() - getId());

		if((*i) != this && nj > ni + 1 && distance < min_distance) {
			jockeyFrom = (*i);
			min_distance = distance;
		}
	}

	// if there is a jockey, then remove that jocky and add it to my own queue
	if(jockeyFrom != NULL) {
		getBoundQueue("MY_QUEUE")->addLast(jockeyFrom->removeBack());
	}
}

Customer* Teller::removeBack() {
	return((Customer*)getBoundQueue("MY_QUEUE")->removeLast());
}

ostream& operator<<(ostream& out, Teller& theTeller) {
	out << "Teller " << theTeller.tellerID << ": Average idle time between customers: "
	    << theTeller.idleTimeStat->getMean() << endl;
	out << "Teller " << theTeller.tellerID << ": Average service time for customers:  "
	    << theTeller.serviceTimeStat->getMean() << endl;
	out << "Teller " << theTeller.tellerID << ": Customer Count:  "
	    << theTeller.customerCount << endl;

	return out;
}
