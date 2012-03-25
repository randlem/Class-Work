#include "Patient.h"

SampST* Patient::totalTimeStat = SimPlus::getInstance()->getSampST();

Patient::Patient()
{
	entryTime = 0;
	startedWaiting = 0;
}

Patient::~Patient()
{
}

void Patient::beginWait( double simTime )
{
	startedWaiting = simTime;
}

double Patient::endWait( double simTime )
{
	return (simTime - startedWaiting);
}

void Patient::enterSystem( double simTime )
{
	entryTime = simTime;
}

void Patient::exitSystem( double simTime )
{
	totalTimeStat->observe( simTime - entryTime );
}

ostream& operator<<( ostream& out, Patient& thePatient )
{
	out << "Average Patient's total turnaround time: "
		<< thePatient.totalTimeStat->getMean() << endl << endl;
	return out;
}
