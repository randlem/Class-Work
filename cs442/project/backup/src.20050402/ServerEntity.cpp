/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

ServerEntity.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of
ServerEntity, see ServerEntity.h.

****************************************************************************/

#include "ServerEntity.h"
#include "SimPlus.h"

// static member intializers
const unsigned short ServerEntity::BEGIN_SERVICE = 0;
const unsigned short ServerEntity::END_SERVICE = 1;
unsigned short ServerEntity::nextPrefix = 100;
unsigned short ServerEntity::prefixIncrement = 100;
NormalRNG* ServerEntity::checkDelay =
	SimPlus::getInstance()->getNormalRNG( 1.0, 0.1 );

ServerEntity::ServerEntity() : Entity()
{
	myPrefix = nextPrefix;
	nextPrefix += prefixIncrement;
	SimPlus::getInstance()->registerServer( this );
}

ServerEntity::~ServerEntity()
{
}

bool ServerEntity::bind( EntityQueue* addMe, string key )
{
	// if the queue is already bound, return false
	map<string, EntityQueue*>::iterator theIterator;
	theIterator = canService.find( key );

	// test whether the iterator is empty
	if( 0 )
		return false;

	// otherwise, insert it and return true
	canService.insert( std::make_pair( key, addMe ) );
	return true;
}

bool ServerEntity::unbind( string key )
{
	if( key == "" )
		return false;
	if( canService.erase( key ) == 0 )
		return false;
	return true;
}

EntityQueue* ServerEntity::getBoundQueue( string key )
{
	if( key == "" )
		return 0;
	map<string, EntityQueue*>::iterator theIterator;
	theIterator = canService.find( key );
	return theIterator->second;
}

Event* ServerEntity::generateEvent( const unsigned short& eventType,
	const double& eventTime )
{
	Event* temp = Entity::generateEvent( eventType, eventTime );
	temp->setEventType( eventType + getPrefix() );
	return temp;
}
