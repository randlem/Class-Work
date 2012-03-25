/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

Entity.h

Object of type Entity represent any object that could possible require control
of the simulation.  Objects that inherit from Entity are automatically assigned
and unsigned int as a unique identifier.  The methods generateEvent and
processEvent are declared virtual to enable polymorphism.  Any object that
inherits from Entity should override these methods if that Object is capable of generating and/or processing events.

METHODS:
--------

Entity()
Sole constructor.  Sets the Entity's ID.

~Entity()
Destructor.

getID() : unsigned int
Returns the ID of the Entity.

virtual operator==(Entity&) : bool
Compares for equality based on ID.

virtual operator!=(Entity&) : bool
Wrapped call to operator==.  Tests for equality based on ID.

virtual operator<(Entity&) : bool
Tests whether the calling Entity's ID is less than the target Entity's ID.

virtual generateEvent(const unsigned short&, const double&) : Event*
Grabs an Event from the kernel's event pool, and sets its attibutes.
First argument determines Event type.  Second argument sets event time.
By default, both owner ID and destination ID are set to the ID of the Entity
calling this method.  The event is automagically inserted into the system
Event list.  A pointer to the Event is returned to allow subclasses to modify
the default behavior.

virtual processEvent(Event*)
Empty function to enable polymorphism.  No default behavior.

reportError( String )
Used to panic the kernel when an Entity is called back with an event it does
not know how to process.

****************************************************************************/

#include <string>
using std::string;

#include "Event.h"

#ifndef ENTITY_H
#define ENTITY_H

class Entity
{
	public:
		// Constructors
		Entity();

		// Destructor
		virtual ~Entity();

		// getters
		inline unsigned int getID() { return myID; };

		// overloaded comparison operators
		// (these 3 are ABSOLUTELY neccessary for STLization)
		virtual bool operator==(Entity&) const;
		virtual bool operator!=(Entity&) const;
		virtual bool operator<(Entity&)  const;

		// some virtual (for polymorphism's sake) default functionality
		virtual Event* generateEvent(const unsigned short&, const double&);
		virtual void processEvent(Event*);

		// these get used by object of type EntityQueue
		Entity* beforeMe;
		Entity* afterMe;

		void reportError( const string& = "Entity" );

	protected:
		unsigned int myID;

	private:
		static unsigned int entitiesDeclared;
};


#endif
