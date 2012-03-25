/****************************************************************************

Nurse.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of Nurse,
see Nurse.h.

****************************************************************************/

#include "Nurse.h"

Nurse::Nurse(double mu, double sigma, EntityQueue* myQueue,
	EntityQueue* myDocQ, EntityQueue* otherDocQ ) : ServerEntity()
{
	SimPlus* handle = SimPlus::getInstance();
	idleTimeStat = handle->getSampST();
	waitTimeStat = handle->getSampST();
	serviceTimeStat = handle->getSampST();

	myServiceTime = handle->getNormalRNG( mu, sigma );

	// for redirection we will use a uniform between zero and one
	myRedirectProb = handle->getUniformRNG( 0.0, 1.0 );

	lastStart = 0;
	lastStop = 0;

	bind( myQueue, "MY_QUEUE" );
	bind( myDocQ, "MY_DOC_QUEUE" );
	bind( otherDocQ, "OTHER_DOC_QUEUE" );

	generateEvent( BEGIN_SERVICE, getCheckDelay() );
}

Nurse::~Nurse()
{
}

Event* Nurse::generateEvent( const unsigned short& eventType, 
	const double& eventTime)
{
	if( eventType != BEGIN_SERVICE && eventType != END_SERVICE )
		return 0;
	return ServerEntity::generateEvent( eventType, eventTime );
}

void Nurse::processEvent( Event* anEvent )
{
	double simTime = SimPlus::getInstance()->getSimTime();
	string sendTo = "MY_DOC_QUEUE";

	if( anEvent->getEventType() == (getPrefix() + BEGIN_SERVICE) )
	{
		// see if there are any patients waiting
		currentPatient = (Patient*)getBoundQueue("MY_QUEUE")->removeFirst();

		// if no patients are waiting, check again in 1 minute
		if( currentPatient == 0 )
			generateEvent( BEGIN_SERVICE, simTime + getCheckDelay() );
		// otherwise go to work
		else
		{
			generateEvent( END_SERVICE, simTime +
				myServiceTime->getRandom() );
			lastStart = simTime;
			idleTimeStat->observe( simTime - lastStop );
			waitTimeStat->observe( currentPatient->endWait( simTime ) );
		}

	}
	else if( anEvent->getEventType() == (getPrefix() + END_SERVICE) )
	{
		// we redirect 15% of the time
		if( myRedirectProb->getRandom() < 0.15 )
			sendTo = "OTHER_DOC_QUEUE";

		// send my patient to the corresponding doc
		getBoundQueue( sendTo )->addLast( currentPatient );
		currentPatient->beginWait( simTime );
		currentPatient = 0;

		lastStop = simTime;
		serviceTimeStat->observe( simTime - lastStart );

		//check for patients again in 1 minute
		generateEvent( BEGIN_SERVICE, simTime + getCheckDelay() );
	}
	else
	{
		reportError( "Nurse" );
	}
	SimPlus::getInstance()->releaseEvent( anEvent );
}

ostream& operator<<( ostream& out, Nurse& theNurse )
{
	out << "Nurse: Average idle time between patients for this nurse: "
		<< theNurse.idleTimeStat->getMean() << endl;
	out << "Nurse: Average service time for patients for this nurse:  "
		<< theNurse.serviceTimeStat->getMean() << endl;
	out << "Nurse: Average patient wait time in queue for this nurse: "
		<< theNurse.waitTimeStat->getMean() << endl << endl;
	return out;
}
