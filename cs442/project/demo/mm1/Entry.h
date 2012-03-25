/****************************************************************************

EntryNode.h

The EntryNode node generates Objects according to the mean interarrival time.

Blatently stolen from the office demo code.

METHODS:
--------

EntryNode( double, int, EntityQueue* )
Sole constructor.  First argument is mean interarrival time.  Second argument
is total number of Objects to be generated.  Third argument is the queue into
which the EntryNode will place the generated Objects.

~EntryNode()
Destructor.

virtual generateEvent( const unsigned short&, const double& ) : Event*
Calls generateEvent in the ServerEntity class.

virtual processEvent( Event* )
Adds a new Object to the associated queue and schedules the next arrival.

****************************************************************************/

#include "SimPlus.h"
#include "Object.h"

#ifndef ENTRYNODE_H
#define ENTRYNODE_H

class EntryNode : public ServerEntity
{
	public:
		EntryNode( double, int, EntityQueue* );
		~EntryNode();

		virtual Event* generateEvent( const unsigned short&, const double& );
		virtual void processEvent( Event* );
		const static unsigned short ARRIVAL;

	protected:
		unsigned short numObjects;
		ExponentialDist* interarrivalTime;
		EntityQueue* entryQueue;
		Object* objectPool;
		unsigned short nextObject;
		unsigned short totalObjects;

	private:
};

#endif
