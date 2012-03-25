/****************************************************************************

EntryNode.h

The EntryNode node generates Patients according to the mean interarrival time.

METHODS:
--------

EntryNode( double, int, EntityQueue* )
Sole constructor.  First argument is mean interarrival time.  Second argument
is total number of Patients to be generated.  Third argument is the queue into
which the EntryNode will place the generated Patients.

~EntryNode()
Destructor.

virtual generateEvent( const unsigned short&, const double& ) : Event*
Calls generateEvent in the ServerEntity class.

virtual processEvent( Event* )
Adds a new Patient to the associated queue and schedules the next arrival.

****************************************************************************/

#include "SimPlus.h"
#include "Patient.h"

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
		unsigned short numPatients;
		ExponentialRNG* interarrivalTime;
		EntityQueue* entryQueue;
		Patient* patientPool;
		unsigned short nextPatient;
		unsigned short totalPatients;

	private:
};

#endif
