/************************************************
*
* Customer.cpp
*
* The implementation of the Customer Entity for the teller simulation.
*
* Written by: Mark Randles, Dan Sinclair
*
************************************************/
#include "Customer.h"

SampST* Customer::totalTimeStat = SimPlus::getInstance()->getSampST();

Customer::Customer()
{
	entryTime = 0;
	startedWaiting = 0;
}

Customer::~Customer() {
}

void Customer::beginWait( double simTime )
{
	startedWaiting = simTime;
}

double Customer::endWait( double simTime )
{
	return (simTime - startedWaiting);
}

void Customer::enterSystem( double simTime )
{
	entryTime = simTime;
}

void Customer::exitSystem( double simTime )
{
	totalTimeStat->observe( simTime, simTime - entryTime );
}

ostream& operator<<( ostream& out, Customer& theCustomer )
{
	out << "Average Customer's total turnaround time: "
		<< theCustomer.totalTimeStat->getMean() << endl << endl;
	return out;
}
