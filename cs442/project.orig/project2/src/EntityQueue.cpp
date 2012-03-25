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
	front = 0;
	back  = 0;
}

EntityQueue::~EntityQueue()
{
}

void EntityQueue::addFirst( Entity* newEntity )
{
	if( newEntity == 0 )
		return;
	if( size == 0 )
		back = newEntity;
	newEntity->beforeMe = 0;
	newEntity->afterMe = front;
	front = newEntity;
	++size;
}

void EntityQueue::addLast( Entity* newEntity )
{
	if( newEntity == 0 )
		return;
	if( size == 0 )
		front = newEntity;
	newEntity->beforeMe = back;
	newEntity->afterMe = 0;
	back = newEntity;
	++size;
}

Entity* EntityQueue::removeFirst()
{
	if( size == 0 )
		return 0;
	Entity* temp = front;
	if( front->afterMe != 0 )
		front = front->afterMe;
	temp->afterMe = 0;
	--size;
	return temp;
}

Entity* EntityQueue::removeLast()
{
	if( size == 0 )
		return 0;
	Entity* temp = back;
	if( back->beforeMe != 0 )
		back = back->beforeMe;
	temp->beforeMe = 0;
	--size;
	return temp;
}

Entity* EntityQueue::removeEntity( unsigned int entityID )
{
	if( size == 0 )
		return 0;
	Entity* temp = find( entityID );
	if( temp != 0 )
	{
		if( temp->beforeMe != 0 )
			temp->beforeMe->afterMe = temp->afterMe;
		if( temp->afterMe != 0 )
			temp->afterMe->beforeMe = temp->beforeMe;
		temp->beforeMe = 0;
		temp->afterMe = 0;
	}
	return temp;
}

Entity* EntityQueue::find( unsigned int entityID ) const
{
	if( size == 0)
		return 0;
	Entity* temp = front;
	while( temp != 0 )
	{
		if( temp->getID() == entityID )
			return temp;
		else
			temp = temp->afterMe;
	}
	return temp;
}
