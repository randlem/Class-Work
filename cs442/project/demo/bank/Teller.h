/************************************************
*
* Teller.h
*
* Teller class definition.
*
* Written by: Mark Randles, Dan Sinclair
*
************************************************/
#include <iostream>
using std::cout;
using std::ostream;
using std::endl;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "SimPlus.h"
#include "Customer.h"

#ifndef TELLER_H
#define TELLER_H

class Teller : public ServerEntity
{
	public:
		Teller( double, double, int, EntityQueue* );
		~Teller();

		virtual Event* generateEvent( const unsigned short&, const double& );
		virtual void processEvent( Event* );

		int getQueueSize();
		inline int getId() { return(tellerID); }

		void jockey();
		Customer* removeBack();

		friend ostream& operator<<( ostream&, Teller& );

	protected:
		int tellerID;
		Customer* currentCustomer;
		static vector<Teller*> tellerList;
		SampST* idleTimeStat;
		SampST* waitTimeStat;
		SampST* serviceTimeStat;
		ExponentialDist* myServiceTime;
		double lastStart;
		double lastStop;
		double closingTime;
		int customerCount;

	private:
};

#endif
