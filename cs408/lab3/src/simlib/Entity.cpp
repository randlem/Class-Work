/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

Entity.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of Entity,
see Entity.h.

****************************************************************************/

#include "Entity.h"
#include "SimPlus.h"

unsigned int Entity::entitiesDeclared = 0;

Entity::Entity()
{
	myID = ++entitiesDeclared;
}

Entity::~Entity()
{
}

bool Entity::operator==( Entity& compareTo ) const
{
	if( myID == compareTo.myID)
		return true;
	return false;
}

bool Entity::operator!=( Entity& compareTo ) const
{
	return !(operator==(compareTo));
}

bool Entity::operator<( Entity& compareTo ) const
{
	if( myID < compareTo.myID )
		return true;
	return false;
}

Event* Entity::generateEvent( const unsigned short& eventType,
	const double& eventTime )
{
	SimPlus* handle = SimPlus::getInstance();
	Event* myEvent = handle->getEvent();
	myEvent->setOwnerID( myID );
	myEvent->setDestination( myID );
	myEvent->setEventType( eventType );
	myEvent->setTimeStamp( eventTime );
	handle->scheduleEvent( myEvent );
	return myEvent;
}

void Entity::processEvent(Event*)
{
}

void Entity::reportError( const string& entityType )
{
	string err = "An object of type \"";
	err += entityType;
	err += "\" attempted to process an event of unknown type.  \n\"";
	err += entityType;
	err += "\" exists in the \"Entity\" inheritance tree.";
	SimPlus::reportError( err );
}
