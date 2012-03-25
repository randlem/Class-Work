/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

Administrator.h

The Administrator node processes incoming Patients and directs them to one
of two Nurse queues with equal probability.

METHODS:
--------

Administrator( double, double, EntityQueue*, EntityQueue*, EntityQueue*)
Sole constructor.  First argument is mean service time.  Second argument is
standard deviation of service time.  Third argument is the queue from which
the Administrator will pull.  Fourth and fifth arguments are the two queues
to which the Administrator can send Patients.  Generates a BEGIN_SERVICE event
to force the node to begin checking for Patients.

~Administrator()
Destructor.

virtual generateEvent( const unsigned short&, const double& ) : Event*
Calls generateEvent in the ServerEntity class.

virtual processEvent( Event* )
Handles event callbacks from the kernel.  For a BEGIN_SERVICE event, the
Administrator grabs the first Patient in its queue.  If the Patient grabbed is
null (zero) a BEGIN_SERVICE event is generated at a random time offset
(a call to getCheckDelay in ServerEntity.)  If the patient is not null, the
Administrator goes to work (schedules an END_SERVICE event at time offset equal
to the next random available from the mean interarrival time.)
For an END_SERVICE event, the Patient is added to the end of one of the two
queues with which the Administrator is associated.

friend operator<<( ostream&, Administrator& ) : ostream&
Allows the Administrator to output its statistics to any ostream, such as
STDOUT or a file.  Returns a reference to the ostream.

****************************************************************************/

#include <iostream>
using std::ostream;
using std::endl;

#include <string>
using std::string;

#include "SimPlus.h"
#include "Patient.h"

#ifndef ADMINISTRATOR_H
#define ADMINISTRATOR_H

class Administrator : public ServerEntity
{
	public:
		Administrator( double, double, EntityQueue*, EntityQueue*,
			EntityQueue* );
		~Administrator();

		virtual Event* generateEvent( const unsigned short&, const double& );
		virtual void processEvent( Event* );

		friend ostream& operator<<( ostream&, Administrator& );

	protected:
		Patient* currentPatient;
		SampST* idleTimeStat;
		SampST* waitTimeStat;
		SampST* serviceTimeStat;
		NormalDist* myServiceTime;
		UniformDist* myRedirectProb;
		double lastStart;
		double lastStop;

	private:
};

#endif
