#include <iostream>
using std::ostream;
using std::endl;

#include "SimPlus.h"

#ifndef PATIENT_H
#define PATIENT_H

class Patient : public Entity
{
	public:
		Patient();
		~Patient();

		static void setEntryQueue( EntityQueue* );

		void beginWait( double );
		double endWait( double );
		void enterSystem( double );
		void exitSystem( double );

		friend ostream& operator<<( ostream&, Patient& );

	protected:
		static SampST* totalTimeStat;

		double entryTime;
		double startedWaiting;

	private:
};

#endif
