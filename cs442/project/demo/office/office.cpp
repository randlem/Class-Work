/****************************************************************************

office.cpp

The driver program for the SimPlus simulation library best practices app.

METHODS:
--------

main() : int
Creates a queueing network representing a healthcare clinic.  A subclass of
ServerEntity, called EntryNode, generates traffic composed of Patient Entities,
arriving according to an Exponential mean interarrival time.  From the
EntryNode, Patients are directed to an Administrator who handles preliminary
paperwork.  The Administrator then distributes the Patients randomly between
two Nurses on approximately a 50/50 basis.  The Nurses then perform
preliminary checks on the Patient.  Each Nurse is associated with a Doctor
and by default sends Patients to that Doctor upon completion of preliminary
exams.  A small portion of the time (15%), a Nurse will redirect its Patients
to the other Doctor.  Each Doctor, upon receiving a Patient, will perform a
more thorough examination of the Patient.  Upon completion of the examination,
a Doctor will sometimes (5% of the time) find it neccessary to send the Patient
to the other doctor Doctor for a second opinion.  If no second opinion is
needed, the Patient is allowed to exit the system.

****************************************************************************/

#include <iostream>
#include <fstream>
#include <iomanip>

#include "SimPlus.h"
#include "EntryNode.h"
#include "Administrator.h"
#include "Nurse.h"
#include "Doctor.h"

using namespace std;

int main()
{
	// take input parameters from a file
	ifstream INFILE;
	INFILE.open( "office.in" );

	double meanIntTime, numPatients, adminMST, adminSTSD;
	double nurseMST, nurseSTSD, doctorMST, doctorSTSD;

	INFILE >> meanIntTime >> numPatients >> adminMST >> adminSTSD;
	INFILE >> nurseMST >> nurseSTSD >> doctorMST >> doctorSTSD;

	INFILE.close();

	// get a handle to the kernel
	SimPlus* handle = SimPlus::getInstance();

	// get some queues from the kernel
	EntityQueue* adminQ = handle->getEntityQueue();
	EntityQueue* nurse1Q = handle->getEntityQueue();
	EntityQueue* nurse2Q = handle->getEntityQueue();
	EntityQueue* doctor1Q = handle->getEntityQueue();
	EntityQueue* doctor2Q = handle->getEntityQueue();

	// instantiate the nodes
	// Because all of the following entities inherit from
	// ServerEntity, it is not neccessary to explicitly
	// register them with the kernel, as the ServerEntity
	// constructor does so automagically.
	EntryNode theFrontDoor( meanIntTime, (int)numPatients, adminQ );
	Administrator admin( adminMST, adminSTSD, adminQ, nurse1Q, nurse2Q );
	Nurse nurse1( nurseMST, nurseSTSD, nurse1Q, doctor1Q, doctor2Q );
	Nurse nurse2( nurseMST, nurseSTSD, nurse2Q, doctor2Q, doctor1Q );
	Doctor doctor1( doctorMST, doctorSTSD, doctor1Q, doctor2Q );
	Doctor doctor2( doctorMST, doctorSTSD, doctor2Q, doctor1Q );

	// 1000 patient simulation
	while( Doctor::patientsProcessed() < 1000 )
	{
		// We don't need to process the events manually, because
		// the kernel handles callbacks.  If we get anything but
		// a zero back from timing, it means that there was no
		// Entity a associated with the current Event.  While this
		// may be perfectly acceptable under some circumstances,
		// this demonstration is written such that all Events should
		// be handled via callbacks.  Thus, if the pointer returned
		// from timing is not zero, we have run into an error.
		if( handle->timing() != 0 )
			handle->reportError( "Unhandled event." );

		//cout << Doctor::patientsProcessed() << " " << setw(7) << handle->getSimTime() << " " << handle->availableEvents() << endl;
	}

	// Output some statistics.  This is not a strict best practice,
	// as writing getters for the statistics associated with each
	// Entity is more robust.  The way
	Patient p;
	cout << endl << endl << p;
	cout << "Administrator:" << endl << admin;
	cout << "Nurse 1:" << endl << nurse1;
	cout << "Nurse 2:" << endl << nurse2;
	cout << "Doctor 1:" << endl << doctor1;
	cout << "Doctor 2:" << endl << doctor2;
	cout << "Simulation ended at time: " << handle->getSimTime() << endl;

	// It is neccessary to explicitly delete the kernel handle to
	// avoid leaking memory.
	delete handle;
	return 0;
}
