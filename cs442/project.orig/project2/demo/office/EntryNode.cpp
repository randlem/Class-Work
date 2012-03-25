/****************************************************************************

EntryNode.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of EntryNode,
see EntryNode.h.

****************************************************************************/

#include "EntryNode.h"

const unsigned short EntryNode::ARRIVAL = 2;

EntryNode::EntryNode( double theta, int patientCapacity,
	EntityQueue* newEntryQueue ) : ServerEntity()
{
	// get a handle to the kernel
	SimPlus* handle = SimPlus::getInstance();

	// mean interarrival time is an exponential distribution
	interarrivalTime = handle->getExponentialRNG( theta );

	// initialize the number of patients to zero
	numPatients = 0;

	// make sure we know where to send new Patients
	entryQueue = newEntryQueue;

	// allocate all of the Patients we need to save us from swap hell
	patientPool = new Patient[patientCapacity];
	nextPatient = 0;
	totalPatients = patientCapacity;

	// get EntryNode started generating Patients
	generateEvent( ARRIVAL, interarrivalTime->getRandom() );
}

EntryNode::~EntryNode()
{
	delete patientPool;
}

Event* EntryNode::generateEvent( const unsigned short& eventType,
	const double& eventTime )
{
	// do a sanity check on the event type we received
	if( eventType != ARRIVAL )
		return 0;
	// then just call base class method
	return ServerEntity::generateEvent( eventType, eventTime );
}

void EntryNode::processEvent( Event* anEvent )
{
	// REMEMBER TO ACCOUNT FOR PREFIXING WHEN HANDLING EVENTS OR THE
	// KERNEL WILL YELL AT YOU!!!! SEE ServerEntity.h FOR MORE INFO!
	if( anEvent->getEventType() == (getPrefix() + ARRIVAL) )
	{
		// get a handle to the kernel
		SimPlus* handle = SimPlus::getInstance();
		double simTime = handle->getSimTime();
		Patient* temp = &patientPool[nextPatient++];
		temp->enterSystem( simTime );
		temp->beginWait( simTime );
		entryQueue->addLast( temp );
		if( nextPatient < totalPatients )
			generateEvent( ARRIVAL, simTime + interarrivalTime->getRandom() );
	}
	else
	{
		reportError( "EntryNode" );
	}
	SimPlus::getInstance()->releaseEvent( anEvent );
}
