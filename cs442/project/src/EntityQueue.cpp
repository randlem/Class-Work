/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

EntityQueue.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of
EntityQueue, see EntityQueue.h.

****************************************************************************/

#include "EntityQueue.h"

EntityQueue::EntityQueue()
{
	size  = 0;
	front = NULL;
	back  = NULL;
}

EntityQueue::~EntityQueue()
{
}

void EntityQueue::addFirst( Entity* newEntity )
{
	if( newEntity == NULL )
		return;
	if( size == 0 )
		back = newEntity;
	newEntity->beforeMe = NULL;
	newEntity->afterMe = front;
	front = newEntity;
	++size;
}

void EntityQueue::addLast( Entity* newEntity )
{
	if( newEntity == NULL )
		return;
	if( size == 0 )
		front = newEntity;
	newEntity->beforeMe = back;
	newEntity->afterMe = NULL;
	back = newEntity;
	++size;
}

Entity* EntityQueue::removeFirst()
{
	if( size == 0 )
		return NULL;
	Entity* temp = front;
	if( front->afterMe != NULL )
		front = front->afterMe;
	temp->afterMe = NULL;
	--size;
	return temp;
}

Entity* EntityQueue::removeLast()
{
	if( size == 0 )
		return NULL;
	Entity* temp = back;
	if( back->beforeMe != NULL )
		back = back->beforeMe;
	temp->beforeMe = NULL;
	--size;
	return temp;
}

Entity* EntityQueue::removeEntity( unsigned int entityID )
{
	if( size == 0 )
		return NULL;
	Entity* temp = find( entityID );
	if( temp != NULL )
	{
		if( temp->beforeMe != NULL )
			temp->beforeMe->afterMe = temp->afterMe;
		if( temp->afterMe != NULL )
			temp->afterMe->beforeMe = temp->beforeMe;
		temp->beforeMe = NULL;
		temp->afterMe = NULL;
	}
	return temp;
}

Entity* EntityQueue::find( unsigned int entityID ) const
{
	if( size == 0)
		return NULL;
	Entity* temp = front;
	while( temp != NULL )
	{
		if( temp->getID() == entityID )
			return temp;
		else
			temp = temp->afterMe;
	}
	return temp;
}
