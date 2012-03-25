/*******************************************************************************
* router.h -- Definition of the Router class
*
* PURPOSE: This class is designed to act as a LP (logical process) of a single
* router and attached clients.
*
* THEORY: Each Router object is ment to encapuslate all the needed functions
* for a single LP.  Each object maintains it's own LVT (local virtual time)
* along with an event list.  Also, each Router object acts like a gateway for
* the attached clients to talk with clients attached to other routers.
* Currently all of these LPs reside in the same thread, but it should be
* possible, with some work, to make it so that each LP talks with each other
* LP though some sort of send/recv mechnism, so that a distributed simulation
* would be possible.  The Client objects attached should need no modification.
*
* IMPLEMENTATION: The Router object uses the STL priority_queue<> class as the
* event list.  The "network" of Router objects is created by the master (in a
* single CPU configuration the main() function), at run-time using the
* attachRouter() method.  The same is true for Clients, only they use the
* attachClient() method.
*
* As the router processes objects, it is it's responsibility to pass a
* structure of data that represents a "packet" of information on the "network".
* This is accomplished by calling the routePacket() function on the object
* that will be recieving Router.  It is up to the recieving router to perform
* the necessairy steps.
*
* To process each packet, the router looks at the address that packet is being
* sent too.  If the packet is ment for one of this Router's clients, then a
* Et_Deliver event is scheduled at the time that the packet will be delivered.
* IF the router is connected directly to the router in the address, then the
* Router will schedule an Et_Route event, and send the packet to that known
* remote Router.  If the packet is not destined for a known location, then the
* router chooses randomly (Uniform distribution) between connected Routers and
* schedules an Et_Route event which in turn will send the packet to the choosen
* Router.
*
* An Et_New event is scheduled by the Router object whenever one of the
* Client's newPacket() returns a time that is < the next scheduled event in the
* Router's event list.  If more then one Client fits that description, then
* the lowest time will be scheduled.  This also negates the need to pump the
* simulation at the start, since the condition is true when the event list is
* empty.
*
* All events act on the top packet in the queue.  Thanks to the causality of the
* simulation this should always be the correct packet.
*******************************************************************************/

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <map>
using std::map;

#include <queue>
using std::queue;
using std::priority_queue;

#include <vector>
using std::vector;

#include <functional>
using std::greater;

#include "base.h"
#include "address.h"
#include "simplus/WeibullDist.h"

#ifndef ROUTER_H
#define ROUTER_H

#include "client.h"

class Router {
public:
	Router(int);
	~Router();

	bool attachRouter(Router*&);
	bool attachClient(Client*&);

	double nextEventTime();
	bool doNextEvent();
	bool routePacket(int, packet*);

	int getId();
	int getClientsCount();
	double getLVT();

	bool registerAddresses(AddressRegister*);

	void dumpStats();

private:
	Router();
	bool scheduleEvent(int,double,packet*,EVENT_type = Et_New);

	map<int, Client*> clients;
	map<int, Router*> routers;
	int 			  id;
	double			  lvt;
	double			  lastRouteTime;
	WeibullDist*	  rngTransmit;
	UniformDist*	  rngRoutePath;

	priority_queue<Event,vector<Event>,greater<Event> > event_list;

};

#endif
