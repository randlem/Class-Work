/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

EventPool.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of EventPool,
see EventPool.h.

****************************************************************************/

#include "EventPool.h"

EventPool::EventPool()
{
	size = 0;
	top = 0;
	bottom = 0;
}

EventPool::EventPool( const unsigned int newSize )
{
	size = 0;
	top = 0;
	bottom = 0;
	for( unsigned int i = 0; i < newSize; i++ )
		release( new Event );
}

EventPool::~EventPool()
{
	while( top != 0 )
	{
		Event* temp = top;
		top = top->nextEvent;
		delete temp;
	}
}

void EventPool::release( Event* newEvent )
{
	if( size == 0 )
		bottom = newEvent;
	else
		newEvent->nextEvent = top;
	top = newEvent;
	++size;
}

Event* EventPool::get()
{
	if( size == 0 )
		return new Event;
	Event* temp = top;
	top = top->nextEvent;
	temp->nextEvent = 0;
	--size;
	return temp;
}

void EventPool::reserve( const unsigned int numEvents )
{
	for( unsigned int i = 0; i < numEvents; i++ )
		release( new Event );
}
