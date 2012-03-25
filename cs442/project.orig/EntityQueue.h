/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

EntityQueue.h

A class composed of a doubly linked list, representing queues capable of 
holding Entity objects.  Implemented as a doubly-linked list.

Care should be taken to avoid inserting a single Entity into multiple queues.
Doing so will likely cause unexpected behavior.

METHODS:
--------

EntityQueue()
Sole constructor.  Sets size, front and back pointers to zero.

~EntityQueue()
Destructor.

addFirst( Entity* )
Adds the sole argument to the front of the queue.

addLast( Entity* )
Adds the sole argument to the end of the queue.

removeFirst()
Removes the last Entity from the queue and returns a pointer to it.  Returns
zero if the queue is empty.

removeLast()
Removes the last Entity from the queue and returns a pointer to it.  Returns
zero if the queue is empty.

removeEntity( unsigned int )
Removes from the queue the Entity with ID equal to the sole argument.  Returns
a pointer to the Entity if it is found, zero if not found.  Operates in O(n).

find( unsigned int )
Finds within the queue the entity with ID equal to the sole argument.  Returns
a pointer to the Entity if it is found, zero if not found.  Leaves the Entity
in the queue.  Operates in O(n).

****************************************************************************/
#include "Entity.h"

#ifndef ENTITYQUEUE_H
#define ENTITYQUEUE_H

class EntityQueue
{
	public:
		EntityQueue();
		~EntityQueue();
		void addFirst( Entity* );
		void addLast( Entity* );
		Entity* removeFirst();
		Entity* removeLast();
		Entity* removeEntity( unsigned int );
		unsigned short getSize() const { return size; };
		Entity* find( unsigned int ) const;

	protected:
		unsigned short size;
		Entity* front;
		Entity* back;

	private:

};

#endif
