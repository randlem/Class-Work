/****************************************************************************

EntryNode.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of EntryNode,
see EntryNode.h.

****************************************************************************/

#include "Entry.h"

const unsigned short EntryNode::ARRIVAL = 2;

EntryNode::EntryNode( double theta, int objectCapacity,
	EntityQueue* newEntryQueue ) : ServerEntity()
{
	// get a handle to the kernel
	SimPlus* handle = SimPlus::getInstance();

	// mean interarrival time is an exponential distribution
	interarrivalTime = handle->getExponentialDist( theta );

	// initialize the number of objects to zero
	numObjects = 0;

	// make sure we know where to send new Objects
	entryQueue = newEntryQueue;

	// allocate all of the Objects we need to save us from swap hell
	objectPool = new Object[objectCapacity];
	nextObject = 0;
	totalObjects = objectCapacity;

	// get EntryNode started generating Objects
	generateEvent( ARRIVAL, interarrivalTime->getRandom() );
}

EntryNode::~EntryNode()
{
	delete [] objectPool;
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
		Object* temp = &objectPool[nextObject++];
		temp->enterSystem( simTime );
		temp->beginWait( simTime );
		entryQueue->addLast( temp );
		if( nextObject < totalObjects )
			generateEvent( ARRIVAL, simTime + interarrivalTime->getRandom() );
	}
	else
	{
		reportError( "EntryNode" );
	}
	SimPlus::getInstance()->releaseEvent( anEvent );
}
