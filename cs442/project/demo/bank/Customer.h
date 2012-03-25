/************************************************
*
* Customer.h
*
* Customer class decleration.
*
* Written by: Mark Randles, Dan Sinclair
*
************************************************/
#include <iostream>
using std::ostream;
using std::endl;

#include "SimPlus.h"

#ifndef CUSTOMER_H
#define CUSTOMER_H

class Customer : public Entity
{
	public:
		Customer();
		~Customer();

		static void setEntryQueue( EntityQueue* );

		void beginWait( double );
		double endWait( double );
		void enterSystem( double );
		void exitSystem( double );

		friend ostream& operator<<( ostream&, Customer& );

	protected:
		static SampST* totalTimeStat;

		double entryTime;
		double startedWaiting;

	private:
};

#endif
