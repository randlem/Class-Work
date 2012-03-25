/****************************************************************************

Nurse.h

The Nurse node processes incoming Patients and directs them to one of two
Doctor queues.  Patients are directed to the primary queue with 85%
probability.

METHODS:
--------

Nurse( double, double, EntityQueue*, EntityQueue*, EntityQueue*)
Sole constructor.  First argument is mean service time.  Second argument is
standard deviation of service time.  Third argument is the queue from which
the Nurse will pull.  Fourth and fifth arguments are the two queues
to which the Nurse can send Patients.  Generates a BEGIN_SERVICE event
to force the node to begin checking for Patients.

~Nurse()
Destructor.

virtual generateEvent( const unsigned short&, const double& ) : Event*
Calls generateEvent in the ServerEntity class.

virtual processEvent( Event* )
Handles event callbacks from the kernel.  For a BEGIN_SERVICE event, the
Nurse grabs the first Patient in its queue.  If the Patient grabbed is
null (zero) a BEGIN_SERVICE event is generated at a random time offset
(a call to getCheckDelay in ServerEntity.)  If the patient is not null, the
Nurse goes to work (schedules an END_SERVICE event at time offset equal
to the next random available from the mean interarrival time.)
For an END_SERVICE event, the Patient is added to the end of one of the two
queues with which the Nurse is associated.

friend operator<<( ostream&, Nurse& ) : ostream&
Allows the Nurse to output its statistics to any ostream, such as STDOUT or a
file.  Returns a reference to the ostream.

****************************************************************************/

#include <iostream>
using std::ostream;
using std::endl;

#include "SimPlus.h"
#include "Patient.h"

#ifndef NURSE_H
#define NURSE_H

#include <string>
using std::string;


class Nurse : public ServerEntity
{
	public:
		Nurse( double, double, EntityQueue*, EntityQueue*, EntityQueue* );
		~Nurse();

		virtual Event* generateEvent( const unsigned short&, const double& );
		virtual void processEvent( Event* );

		friend ostream& operator<<( ostream&, Nurse& );

	protected:
		Patient* currentPatient;
		SampST* waitTimeStat;
		SampST* idleTimeStat;
		SampST* serviceTimeStat;
		NormalDist* myServiceTime;
		UniformDist* myRedirectProb;
		double lastStart;
		double lastStop;

	private:
};

#endif
