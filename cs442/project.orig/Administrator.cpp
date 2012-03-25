/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

Doctor.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of Doctor,
see Doctor.h.

****************************************************************************/

#include "Administrator.h"

Administrator::Administrator(double mu, double sigma, EntityQueue* myQ,
	EntityQueue* n1Q, EntityQueue* n2Q ) : ServerEntity()
{
	SimPlus* handle = SimPlus::getInstance();
	idleTimeStat = handle->getSampST();
	waitTimeStat = handle->getSampST();
	serviceTimeStat = handle->getSampST();

	myServiceTime = handle->getNormalRNG( mu, sigma );

	// for redirection we will use a uniform between zero and one
	myRedirectProb = handle->getUniformRNG( 0.0, 1.0 );
	myRedirectProb->seedRand( 1235312562 );

	lastStart = 0;
	lastStop = 0;

	bind( myQ, "MY_QUEUE" );
	bind( n1Q, "NURSE_ONE_QUEUE" );
	bind( n2Q, "NURSE_TWO_QUEUE" );

	generateEvent( BEGIN_SERVICE, getCheckDelay() );
}

Administrator::~Administrator()
{
}

Event* Administrator::generateEvent( const unsigned short& eventType, 
	const double& eventTime)
{
	if( eventType != BEGIN_SERVICE && eventType != END_SERVICE )
		return 0;
	return ServerEntity::generateEvent( eventType, eventTime );
}

void Administrator::processEvent( Event* anEvent )
{
	double simTime = SimPlus::getInstance()->getSimTime();

	if( anEvent->getEventType() == (getPrefix() + BEGIN_SERVICE) )
	{
		// see if there are any patients waiting
		currentPatient = (Patient*)getBoundQueue("MY_QUEUE")->removeFirst();

		// if no patients are waiting, check again in 1 minute
		if( currentPatient == 0 )
		{
			generateEvent( BEGIN_SERVICE, simTime + getCheckDelay() );
		}
		// otherwise go to work
		else
		{
			generateEvent( END_SERVICE, simTime +
				myServiceTime->getRandom() );
			lastStart = simTime;
			idleTimeStat->observe( simTime - lastStop );
		}

	}
	else if( anEvent->getEventType() == (getPrefix() + END_SERVICE) )
	{
		string sendTo = "NURSE_ONE_QUEUE";
		// we redirect 50% of the time
		if( myRedirectProb->getRandom() < 0.5 )
			sendTo = "NURSE_TWO_QUEUE";

		// send my patient to the corresponding doc
		getBoundQueue( sendTo )->addLast( currentPatient );
		currentPatient = 0;

		lastStop = simTime;
		serviceTimeStat->observe( simTime - lastStart );

		//check for patients again in 1 minute
		generateEvent( BEGIN_SERVICE, simTime + getCheckDelay() );
	}
	else
	{
		reportError( "Administrator" );
	}
	SimPlus::getInstance()->releaseEvent( anEvent );
}

ostream& operator<<( ostream& out, Administrator& theAdministrator )
{
	out << "Administrator: Average idle time between patients: "
	    << theAdministrator.idleTimeStat->getMean() << endl;
	out << "Administrator: Average service time for patients:  "
	    << theAdministrator.serviceTimeStat->getMean() << endl;
	out << "Administrator: Average patient wait time in queue: "
	    << theAdministrator.waitTimeStat->getMean() << endl << endl;
	return out;
}

