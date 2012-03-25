/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

Event.h

Event object are used to control the flow of a system.

METHODS:
--------

Event()
Default constructor.  Initializes all data member to zero, and set the Event's
ID.

Event( double, unsigned int, unsigned int, unsigned short )
Constructor.  The first argument is used to set the timeStamp, the second sets
the ownerID, the third the destinationID, and the fourth sets the eventType.
The Event's ID is set automatically.

~Event()
Destructor.

operator==( const Event& theEvent ) : bool
Tests for equality based on Event ID.


operator!=( const Event& theEvent ) : bool
Wrapped call to operator==.  Tests for inequality based on EventID.

operator<( const Event& theEvent ) : bool
Tests whether the calling event has a smaller timeStamp (i.e. is scheduled to
occur earlier) than the other object.

operator>( const Event& theEvent ) : bool
Tests whether the calling event has a larger timeStamp (i.e. is scheduled to
occur later) than the other object.

reset()
Reinitializes the data members of the Event to zero.

****************************************************************************/

#ifndef EVENT_H
#define EVENT_H

class Event
{

	public:
		// constructors
		Event();
		Event( double, unsigned int, unsigned int, unsigned short );

		// Destructor.
		~Event();

		// Getters
		double getTimeStamp()           { return timeStamp;       };
		unsigned int getOwnerID()       { return ownerID;         };
		unsigned int getDestination()   { return destinationID;   };
		unsigned short getEventType()   { return eventType;       };
		unsigned int getID()            { return myID;            };

		// Setters
		void setTimeStamp(double newTimeStamp)
			{ timeStamp = newTimeStamp; };
		void setOwnerID(unsigned int newOwnerID)
			{ ownerID = newOwnerID;   };
		void setDestination(unsigned int newDestination)
			{ destinationID = newDestination;   };
		void setEventType(unsigned short newEventType)
			{ eventType = newEventType; };

		// overloaded comparison operators 
		bool operator==(const Event&) const;
		bool operator!=(const Event&) const;
		bool operator<(const Event&) const;
		bool operator>(const Event&) const;

		// Event junk used by the event allocation pool
		Event* nextEvent;

		// Clear state information for Event object
		void reset();

		// Manipulate cancellation status of Event
		void cancel()
			{ cancelled = true; }
		const bool isCancelled() const
			{ return cancelled; }

	protected:
		double timeStamp;            // Time at which this event is scheduled
								     // to occur

		unsigned int ownerID;        // ID of entity that created this event

		unsigned int destinationID; // ID of entity where this event will take
									// place

		unsigned short eventType;   // Defined in entity that generated event
		bool cancelled;

	private:
		static unsigned int eventsDeclared;
		unsigned int myID;
};

#endif
