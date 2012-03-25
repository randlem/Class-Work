// EventList.h
// Scott Harper, Tom Mancine, Ryan Scott
//
//
// EventList
// ---------
// An abstract base class defining the interface for the various
// event queues provided by SimPlus.
//
// EventListException
// ------------------
// Defines the exceptions thrown by the event queues from
// SimPlus.

#ifndef EVENTLIST
#define EVENTLIST

class EventList
{
	public:

		// constructor (not sure why we must provide this...)
		EventList(){}

		// destructor must be virtual since we subclass EventList
		virtual ~EventList(){}

		// minimum set of methods subclasses must implement
		virtual Event * get() = 0;
		virtual bool put(Event *) = 0;
		virtual bool resize(unsigned short newSize) = 0;
		virtual bool cancelNext(unsigned short evtType) = 0;
};

class EventListException
{
	public:
		// Construct exception object with error message
		EventListException(char * mesg)
			{ message = mesg; }

		// Destroy exception object
		virtual ~EventListException(){}

		// All we can do is get the error message
		virtual char * getMessage()
			{ return message; }
	protected:
		char * message;
};

#endif
