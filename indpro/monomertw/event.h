#include "latprim.h"

#ifndef EVENT_H
#define EVENT_H

enum EventType {eventDeposition, eventDiffusion};

class Event {
public:
	Event(site* oldSite, site* newSite, double time, bool isLocal, EventType eventType,/* Direction dir,*/ int listIndex) {
		this->oldSite = oldSite;
		this->newSite = newSite;
		this->time = time;
		this->isLocal = isLocal;
		this->eventType = eventType;
		/*this->dir = dir;*/
		this->listIndex = listIndex;
	}

	Event(site* newSite, double time, bool isLocal, EventType eventType/*, Direction dir, int listIndex*/) {
		this->newSite = newSite;
		this->time = time;
		this->isLocal = isLocal;
		this->eventType = eventType;
		/*this->dir = dir;*/
		/*this->listIndex = listIndex;*/
	}

	site* oldSite;       // the old site (diffusion event)
	site* newSite;       // the new site (diffusion event) or the site for a deposition event
	double time;         // event time
	bool isLocal;        // true if a local event
	bool isBoundry;      // true if a boundry event
	EventType eventType; // type of event
//	Direction dir;       // important if it's a boundry event, which neighbor to send it too
	int listIndex;       // index of the site in the list
private:
	Event() { ; } // supress creating an event with a default constructor
};

#endif


