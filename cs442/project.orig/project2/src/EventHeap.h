// EventHeap.h
// Scott Harper, Tom Mancine, Ryan Scott
//
// EventHeap
// ---------
// A heap data structure for storing SimPlus Events.
// Implemented as an in-place heap; permits insertion,
// retrieval, cancellation, and resizing.
// Acts like a priority queue ordered by Event::timeStamp.

#include "Event.h"
#include "EventList.h"

#ifndef EVENTHEAP
#define EVENTHEAP

class EventHeap : public EventList
{
	public:
		// constructor; default intial capacity is 128
		EventHeap(unsigned short initCapacity=128);

		// destructor is virtual to permit subclassing
		virtual ~EventHeap();

		// fetch the lowest-stamped Event in the heap
		// that has not been cancelled
		virtual Event * get();

		// put an Event into the heap
		virtual bool put(Event *);

		// explicitly resize heap to newSize; heap auto-resizes
		// by using this method if it gets full, and the user
		// may invoke it as an efficiency measure
		virtual bool resize(unsigned short newSize);

		// cancel next event of type evtType; does not remove
		// Event from heap, but does flag it as cancelled
		virtual bool cancelNext(unsigned short evtType);

	protected:

		// private members
		unsigned short capacity, size;
		Event ** heap;
};

#endif
