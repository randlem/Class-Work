/****************************************************************************

Scott Harper, Tom Mancine, Ryan Scott

Doctor.h

The Doctor node processes incoming Patients and either removes them from the
system or sends them to another Doctor's queue (5% probability).

METHODS:
--------

Doctor( double, double, EntityQueue*, EntityQueue*)
Sole constructor.  First argument is mean service time.  Second argument is
standard deviation of service time.  Third argument is the queue from which
the Doctor will pull.  Fourth argument is the queue of the other Doc
to which the Doctor can send Patients.  Generates a BEGIN_SERVICE event
to force the node to begin checking for Patients.

~Doctor()
Destructor.

virtual generateEvent( const unsigned short&, const double& ) : Event*
Calls generateEvent in the ServerEntity class.

virtual processEvent( Event* )
Handles event callbacks from the kernel.  For a BEGIN_SERVICE event, the
Doctor grabs the first Patient in its queue.  If the Patient grabbed is
null (zero) a BEGIN_SERVICE event is generated at a random time offset
(a call to getCheckDelay in ServerEntity.)  If the patient is not null, the
Doctor goes to work (schedules an END_SERVICE event at time offset equal
to the next random available from the mean interarrival time.)
For an END_SERVICE event, the Patient is either removed from the system or is
added to the end of the other Doctor's queue (with 5% probability).

friend operator<<( ostream&, Doctor& ) : ostream&
Allows the Doctor to output its statistics to any ostream, such as
STDOUT or a file.  Returns a reference to the ostream.

****************************************************************************/

#include <iostream>
using std::iostream;
using std::cout;
using std::endl;

#include <string>
using std::string;

#include "SimPlus.h"
#include "Patient.h"

#ifndef DOCTOR_H
#define DOCTOR_H

class Doctor : public ServerEntity
{
	public:
		Doctor( double, double, EntityQueue*, EntityQueue* );
		~Doctor();

		virtual Event* generateEvent( const unsigned short&, const double& );
		virtual void processEvent( Event* );
		static unsigned short patientsProcessed() { return numPatients; };

		friend ostream& operator<<( ostream&, Doctor& );

	protected:
		static unsigned short numPatients;
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
