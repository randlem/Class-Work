/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

Doctor.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of Doctor,
see Doctor.h.

****************************************************************************/

#include "Doctor.h"

unsigned short Doctor::numPatients = 0;

Doctor::Doctor( double mu, double sigma, EntityQueue* myQ,
	EntityQueue* otherQ ) : ServerEntity()
{
	SimPlus* handle = SimPlus::getInstance();
	idleTimeStat = handle->getSampST();
	waitTimeStat = handle->getSampST();
	serviceTimeStat = handle->getSampST();

	// this doctor's service time is a normal distribution with a mean of
	// fifteen minutes and a standard deviation of five minutes
	myServiceTime = handle->getNormalRNG( mu, sigma );
//	myServiceTime = handle->getNormalRNG( mu, sigma, RawRNG::Net);

	// for redirection we will use a uniform between zero and one
	myRedirectProb = handle->getUniformRNG( 0.0, 1.0 );

	lastStart = 0;
	lastStop = 0;

	bind( myQ, "MY_QUEUE" );
	bind( otherQ, "OTHER_DOC_QUEUE" );

	generateEvent( BEGIN_SERVICE, getCheckDelay() );
}

Doctor::~Doctor()
{
}

Event* Doctor::generateEvent( const unsigned short& eventType, 
	const double& eventTime)
{
	if( eventType != BEGIN_SERVICE && eventType != END_SERVICE )
		return 0;
	return ServerEntity::generateEvent( eventType, eventTime );
}

void Doctor::processEvent( Event* anEvent )
{
	double simTime = SimPlus::getInstance()->getSimTime();

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
		// we redirect 5% of the time
		if( myRedirectProb->getRandom() < 0.05 )
		{
			getBoundQueue( "OTHER_DOC_QUEUE" )->addLast( currentPatient );
			currentPatient->beginWait( simTime );
		}
		// otherwise we kick the patient out of the system
		else
		{
			++numPatients;
			currentPatient->exitSystem( simTime );
			//delete currentPatient;
		}

		// send my patient to the corresponding doc
		currentPatient = 0;

		lastStop = simTime;
		serviceTimeStat->observe( simTime - lastStart );

		//check for patients again in 1 minute
		generateEvent( BEGIN_SERVICE, simTime + getCheckDelay() );
	}
	else
	{
		reportError( "Doctor" );
	}
	SimPlus::getInstance()->releaseEvent( anEvent );
}

ostream& operator<<( ostream& out, Doctor& theDoctor )
{
	out << "Doctor: Average idle time between patients for this doctor: "
	    << theDoctor.idleTimeStat->getMean() << endl;
	out << "Doctor: Average service time for patients for this doctor:  "
	    << theDoctor.serviceTimeStat->getMean() << endl;
	out << "Doctor: Average patient wait time in queue for this doctor: "
	    << theDoctor.waitTimeStat->getMean() << endl << endl;
	return out;
}

