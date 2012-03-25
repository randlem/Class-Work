/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

ServerEntity.h

ServerEntity is a subclass of Entity designed to represent Entities that are
capable of performing service on/for other Entities moving through the system.

METHODS:
--------

ServerEntity()
Assigns this ServerEntity its numeric prefix and registers it with the kernel
for Event callbacks.

virtual ~ServerEntity()
Virtual destructor.

bind( EntityQueue*, string ) : bool
Binds the specified EntityQueue to the current ServerEntity.  Pointers to
EntityQueues are stored in associative arrays (STL maps) with key equal to
the second argument.  A ServerEntity may be bound to more than one EntityQueue
and an EntityQueue may be bound by more than one ServerEntity.  Binding only
implies that a ServerEntity may insert into/extract from an EntityQueue.
Returns true if the insertion is successfule, false otherwise.

unbind( string ) : bool
Unbinds the queue with key equal to sole argument.  Returns true if the queue
is unbound, false if it is not bound to begin with.

getBoundQueue( string ) : EntityQueue*
Returns the queue associated with the sole argument.

virtual generateEvent( const unsigned short&, const double& ) : Event* 
Wraps call to Entity::generateEvent.  Adds this ServerEntity's prefix to the
Event's eventType.  Care should be taken to account for prefixes in
subclass Event processing.

getCheckDelay() : double
Returns a normally distributed random number with mean 1.0 and standard
deviation 0.1.  The preferred method for performing service is to generate and
process BEGIN_SERVICE events at offset equal to checkDelay->getRandom() until
an Entity is available for service.

getPrefix() : unsigned short
Returns this ServerEntity's prefix.

****************************************************************************/

#include <string>
using std::string;

#include <map>
using std::map;
using std::pair;

#include "Entity.h"
#include "EntityQueue.h"
#include "NormalRNG.h"

#ifndef SERVERENTITY_H
#define SERVERENTITY_H

class ServerEntity : public Entity
{
	public:
		ServerEntity();
		virtual ~ServerEntity();
		bool bind( EntityQueue*, string );
		bool unbind( string );
		EntityQueue* getBoundQueue( string );

		const static unsigned short BEGIN_SERVICE;
		const static unsigned short END_SERVICE;

		virtual Event* generateEvent( const unsigned short&, const double& );

		double getCheckDelay()     { return checkDelay->getRandom(); };
		unsigned short getPrefix() { return myPrefix;                };

	protected:

	private:
		static NormalRNG* checkDelay;
		static unsigned short nextPrefix;
		static unsigned short prefixIncrement;
		unsigned short myPrefix;
		map<string, EntityQueue*> canService;
};

#endif
