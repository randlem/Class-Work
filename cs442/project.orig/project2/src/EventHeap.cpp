#include "SimPlus.h"
#include "EventHeap.h"

#include <cstring>
#include <strings.h>

EventHeap::EventHeap(unsigned short initCapacity)
{
	heap = NULL; size=0;
	if(!initCapacity)
		throw EventListException("Invalid event heap parameters.");
	if(!resize(initCapacity))
		throw EventListException("Could not construct event heap.");
}

EventHeap::~EventHeap()
{
	if(heap) delete[] heap;
}

bool EventHeap::resize(unsigned short newSize)
{
	Event** newheap;

	if(newSize<size)
		return false;

	if(!(newheap = new Event*[1+newSize]))
		return false;

	bzero(newheap,newSize*sizeof(Event *));

	if(heap)
	{
		bcopy(heap,newheap,(1+size)*sizeof(Event **));
		delete[] heap;
	}

	heap=newheap;
	capacity=newSize;

	return true;
}

Event * EventHeap::get()
{
	if(!size)
		return NULL;

	unsigned int n, u=1, v;
	Event * tmpE, *retVal, *tmp;

	retVal=heap[1];
	tmpE=heap[n=size--];

	while(u<=n/2)
	{
		v=2*u;

		if(v<n && (*heap[v]>*heap[v+1]))
			v++;

		if(*heap[v] > *tmpE)
			break;

		tmp=heap[u];
		heap[u]=heap[v];
		heap[v]=tmp;
		u=v;
	}

	heap[u]=tmpE;

	if(retVal->isCancelled())
	{
		SimPlus::getInstance()->releaseEvent(retVal);
		return get();
	}

	return retVal;
}

bool EventHeap::put(Event * evt)
{
	unsigned int index;
	Event * tmpE;

	if(size==capacity){
		if(!resize(2*capacity))
			return false;
	}

	index=++size;

	while(index > 1 && (*heap[index/2] > *evt))
	{
		tmpE=heap[index/2];
		heap[index/2]=heap[index];
		heap[index]=tmpE;
		index=index/2;
	}

	heap[index]=evt;

	return true;
}

bool EventHeap::cancelNext(unsigned short evtType)
{
	Event* target = NULL;

	for(int i=1; i<=size; i++)
		if(heap[i]->getEventType() == evtType)
			if(!target || *target>*heap[i])
				target=heap[i];

	if(!target) return false;

	target->cancel();
	return true;
}
