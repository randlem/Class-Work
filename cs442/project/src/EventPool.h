/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

EventPool.h

EventPool is used to speed up the SimPlus kernel by reducing the costly
overhead associated with liberal use of the new operator.  By pre-allocating
a stack of Event objects, we save the user from using the new operator by
allowing Events to be checked in and checked out from the pool when needed.
The EventPool will somtimes need to use the new operator itself, but the
Pool structure's pre-allocation minimizes that need.

METHODS:
--------

EventPool()


EventPool( const unsigned int )
Create an empty EventPool.

~EventPool()
Deletes all Events in the pool.

release( Event* )
Returns the first argument to the EventPool.

get() : Event*
Returns the top element of the EventPool.

reserve( const unsigned int )
Allocates a number of additional Events equal to the sole argument.

getSize() : unsigned int
Returns the number of Events currently in the pool.

****************************************************************************/

#include "Event.h"

#ifndef EVENTPOOL_H
#define EVENTPOOL_H

class EventPool
{
	public:
		EventPool();
		EventPool( const unsigned int );
		~EventPool();
		void release( Event* );
		Event* get();
		void reserve( const unsigned int );
		unsigned int getSize() const { return size; };

	protected:

	private:
		unsigned int size;
		Event* top;
		Event* bottom;
};

#endif
