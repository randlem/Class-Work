/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

Event.cpp

The documentation within this file is sparse, and is only intended to provide
an overview of coding practices.  For a more detailed description of Event,
see Event.h.

****************************************************************************/

#include "Event.h"

unsigned int Event::eventsDeclared = 0;

Event::Event()
{
	timeStamp     = 0;
	ownerID       = 0;
	destinationID = 0;
	eventType     = 0;
	nextEvent     = 0;
	myID          = ++eventsDeclared;
	cancelled     = false;
}

Event::Event(double newTimeStamp, unsigned int newOwnerID,
	unsigned int newDestination, unsigned short newEventType )
{
	timeStamp     = newTimeStamp;
	ownerID       = newOwnerID;
	destinationID = newDestination;
	eventType     = newEventType;
	nextEvent     = 0;
	myID          = ++eventsDeclared;
	cancelled     = false;
}

Event::~Event()
{
}

bool Event::operator==(const Event& theEvent) const
{
	if( myID == theEvent.myID )
		return true;
	return false;
}

bool Event::operator!=(const Event& theEvent) const
{
	return !(operator==(theEvent));
}

bool Event::operator<(const Event& theEvent) const
{
	if( timeStamp < theEvent.timeStamp )
		return true;
	return false;
}

bool Event::operator>(const Event& theEvent) const
{
	if( timeStamp > theEvent.timeStamp )
		return true;
	return false;
}

void Event::reset()
{
	timeStamp     = 0;
	ownerID       = 0;
	destinationID = 0;
	eventType     = 0;
	nextEvent     = 0;
	cancelled     = false;
}
