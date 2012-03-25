/****************************************************************************

SimPlus.h

Also know as "The Kernel," SimPlus is a C++ class that encapsulates most of the
functionality required for small to medium scale simulation projects.  Manages
the creation and distribution of simulation-related objects.  SimPlus keeps
track of those objects so that it can clean up after itself.  SimPlus is
loosely based on the simlib.c code.

SimPlus is implemented using the "Singleton" design pattern.  This means that
within a given execution, only one instance of the Kernel object may be
created.  This is ensured by declaring the constructor, copy constructor, and
overloaded assignment operator to be protected.

PUBLIC METHODS:
---------------

static getInstance() : SimPlus*
Returns a pointer to the current instance of SimPlus.  Creates a new instance
and returns a pointer to it if this is the first call.

~SimPlus()
Explcitly deletes all of the object references handed out by calls to the
various getXXX methods.

static reportError( string )
Panics the kernel, prints the error message contained in the sole argument and
exits with an error code of 1.  To be used in the case of non-recoverable
errors only.

timing() : Event*
Processes the next event contained in the event list.  If the Entity where the
event is scheduled to take place has registered itself with the system (default
behavior for objects of type ServerEntity), timing calls the processEvent
method for that Entity and returns zero.  If the target Entity is not
registered with the kernel, a pointer to the Event object is returned so that
the Event may be processed manually.

registerServer( ServerEntity* )
Registers the sole argument with the kernel for event callbacks.

scheduleEvent( Event* )
Inserts the sole argument into the system's event list.  Calls reportError if
the event list cannot or will not accept more Events.

cancelEventType( const unsigned short& ) : bool
Cancels the next event with type equal to its sole argument.  Operates in O(n)
time.  Returns true if the event is cancelled, false otherwise.

cancelEventID( const unsigned int& ) : bool
Cancels a specific event with eventID equals to its sole argument.  Operates in
O(n) time.  Returns true if the event is cancelled, false otherwise.

getEvent() : Event*
To save time once a simulation has begun executing, an initial pool of Event
objects is allocated.  The getEvent method returns a handle to the top Event in
the pool if the pool has any events in it, and returns a new Event if the pool
is empty.

releaseEvent( Event* )
Calls the reset method of its sole parameter and then inserts it into the
Event pool for reuse.

expandEventPool( const unsigned short& )
Causes the EventPool to allocate a number of new events equal to its sole
argument (usually in anticipation of a large number of new events.)

availableEvents() : unsigned int
Returns the number of Event objects currently accounted for in the EventPool.

getEntityQueue() : EntityQueue*
Allocates a new EntityQueue for the user, saves a reference to it for cleanup,
and returns a pointer to it.

getSampST() : SampST*
Allocates a new SampST for the user, saves a reference to it for cleanup,
and returns a pointer to it.

getExponentialRNG( const double&, unsigned short ) : ExponentialRNG*
Allocates a new ExponentialRNG for the user, saves a reference to it for
cleanup, and returns a pointer to it.  The second argument indicates the type
of RawRNG to be used in the ExponentialRNG being created.  The default is
RawRNG::Local, which is seeded.  Another possibility is RawRNG::Net, which
retrieves truly random numbers via an HTTPSocket from random.org.
The first argument specifies the mean of the numbers generated.

getNormalRNG( const double&, const double&, unsigned short ) : NormalRNG*
Allocates a new NormalRNG for the user, saves a reference to it for
cleanup, and returns a pointer to it.  The last argument indicates the type
of RawRNG to be used in the NormalRNG being created.  The default is
RawRNG::Local, which is seeded.  Another possibility is RawRNG::Net, which
retrieves truly random numbers via an HTTPSocket from random.org.
The first argument specifies the mean of the numbers generated.  The second
specifies the standard deviation.

getUniformRNG( const double&, unsigned short ) : UniformRNG*
Allocates a new UniformRNG for the user, saves a reference to it for
cleanup, and returns a pointer to it.  The second argument indicates the type
of RawRNG to be used in the UniformRNG being created.  The default is
RawRNG::Local, which is seeded.  Another possibility is RawRNG::Net, which
retrieves truly random numbers via an HTTPSocket from random.org.
The first argument specifies the lower bound of the numbers generated.  The
second specifies the upper bound.

getSimTime() : double
Returns the current simulation time.

****************************************************************************/

#ifndef SIMPLUS_H
#define SIMPLUS_H

#include "SampST.h"

#include "Event.h"
#include "EventPool.h"
#include "EventList.h"
#include "EventHeap.h"

#include "ServerEntity.h"
#include "EntityQueue.h"

#include "RNGFactory.h"
#include "ExponentialDist.h"
#include "NormalDist.h"
#include "UniformDist.h"

#include <stack>
#include <string>
#include <map>

using std::stack;
using std::string;
using std::map;
using std::pair;

class SimPlus
{
	public:
		static SimPlus* getInstance();
		static void reportError( const string& );
		~SimPlus();

		Event* timing();
		void scheduleEvent( Event* );
		void registerServer( ServerEntity* );
		bool cancelEventType( const unsigned short& );
		bool cancelEventID( const unsigned int& );
		Event* getEvent();
		void releaseEvent( Event* );
		void expandEventPool( const unsigned short& );
		unsigned int availableEvents();

		EntityQueue* getEntityQueue();
		SampST* getSampST();

		ExponentialDist* getExponentialDist(const double&, const RNGFactory::RNGType = (RNGFactory::RNGType)0);
		NormalDist* getNormalDist(const double&, const double&, const RNGFactory::RNGType = (RNGFactory::RNGType)0);
		UniformDist* getUniformDist(const double&, const double&, const RNGFactory::RNGType = (RNGFactory::RNGType)0);

		inline double getSimTime() { return simTime; };

	protected:
		SimPlus();
		SimPlus( const SimPlus& );
		SimPlus& operator=( const SimPlus& );

	private:
		static SimPlus* theKernel;

		double simTime;

		EventPool theEventPool;
		EventList* theEventList;
		stack<RawDist*> randomStack;
		stack<SampST*> sampleStack;
		stack<EntityQueue*> queueStack;
		map<unsigned int, ServerEntity*> serverMap;
};

#endif
