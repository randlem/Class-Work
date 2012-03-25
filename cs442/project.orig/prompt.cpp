#include <iostream>
#include <fstream>

#include "SimPlus.h"
#include "EntryNode.h"
#include "Administrator.h"
#include "Nurse.h"
#include "Doctor.h"

using namespace std;

int main()
{
	ifstream INFILE;
	INFILE.open( "office.in" );

	double meanIntTime, numPatients, adminMST, adminSTSD;
	double nurseMST, nurseSTSD, doctorMST, doctorSTSD;

	INFILE >> meanIntTime >> numPatients >> adminMST >> adminSTSD;
	INFILE >> nurseMST >> nurseSTSD >> doctorMST >> doctorSTSD;

	INFILE.close();

	cout << "MIT ? >";
	cin >> meanIntTime;

	SimPlus* handle = SimPlus::getInstance();

	EntityQueue* adminQ = handle->getEntityQueue();
	EntityQueue* nurse1Q = handle->getEntityQueue();
	EntityQueue* nurse2Q = handle->getEntityQueue();
	EntityQueue* doctor1Q = handle->getEntityQueue();
	EntityQueue* doctor2Q = handle->getEntityQueue();

	EntryNode theFrontDoor( meanIntTime, numPatients, adminQ );
	Administrator admin( adminMST, adminSTSD, adminQ, nurse1Q, nurse2Q );
	Nurse nurse1( nurseMST, nurseSTSD, nurse1Q, doctor1Q, doctor2Q );
	Nurse nurse2( nurseMST, nurseSTSD, nurse2Q, doctor2Q, doctor1Q );
	Doctor doctor1( doctorMST, doctorSTSD, doctor1Q, doctor2Q );
	Doctor doctor2( doctorMST, doctorSTSD, doctor2Q, doctor1Q );

	// 1000 patient simulation
	while( Doctor::patientsProcessed() < 1000 )
	{
		handle->timing();
	}

	Patient p;
	cout << endl << endl << p;
	cout << "Administrator:" << endl << admin;
	cout << "Nurse 1:" << endl << nurse1;
	cout << "Nurse 2:" << endl << nurse2;
	cout << "Doctor 1:" << endl << doctor1;
	cout << "Doctor 2:" << endl << doctor2;
	cout << "Simulation ended at time: " << handle->getSimTime() << endl;

	delete handle;
	return 0;
}
