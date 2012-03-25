/************************************************
*
* Door.h
*
* Door class definition.
*
* Written by: Mark Randles, Dan Sinclair
*
************************************************/
#include <iostream>
using std::cout;
using std::endl;

#include <vector>
using std::vector;

#include "SimPlus.h"
#include "Customer.h"

#ifndef DOOR_H
#define DOOR_H

class Door : public ServerEntity
{
	public:
		Door(double, double, vector<EntityQueue*>& tellerQueue);
		~Door();

		virtual Event* generateEvent(const unsigned short&, const double&);
		virtual void processEvent(Event*);
		const static unsigned short ARRIVAL;

	protected:
		ExponentialDist* interarrivalTime;
		vector<EntityQueue*> tellerQueue;
		vector<Customer*> customerPool;
		double doorCloseTime;

	private:
};

#endif
