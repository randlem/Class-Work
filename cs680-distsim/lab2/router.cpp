#include "router.h"

Router::Router(int id) {
	this->id 			= id;
	this->lvt			= 0.0;
	this->lastRouteTime = 0.0;
	this->rngTransmit	= new WeibullDist(2.0,1.0);
}

Router::~Router() {
	if(rngTransmit != NULL)
		delete this->rngTransmit;
}

bool Router::attachRouter(Router*& r) {
	// add a pointer to the router object to the list
	routers[r->getId()] = r;

	// clear the uniform generatior
	if(rngRoutePath != NULL)
		delete rngRoutePath;
}

bool Router::attachClient(Client*& c) {
	// add a pointer to the client object to the list
	clients[c->getId()] = c;

	// let the client know who it's parent is
	c->attachRouter(this);
}

double Router::nextEventTime() {
	double nextTime = -1.0;

	// if there isn't any event in the list, look to see when the next new packet
	// is an either return -1.0 or the time of that new packet
	if(event_list.empty()) {
		for(map<int, Client*>::iterator i=clients.begin(); i != clients.end(); ++i) {
			Client* c = (*i).second;
			double t = lvt + c->nextNewPacket();
			if(nextTime < 0 || t < nextTime)
				nextTime = t;
		}

		return nextTime;
	}

	Event e = event_list.top();

	// find the minimum time, either the next new packet or the top event
	nextTime = e.timestamp;
	for(map<int, Client*>::iterator i=clients.begin(); i != clients.end(); ++i) {
		Client* c = (*i).second;
		double t = lvt + c->nextNewPacket();
		if(t < nextTime)
			nextTime = t;
	}

	// otherwise return the timestamp of the top event
	return nextTime;
}

bool Router::doNextEvent() {
	Event e;
	Client* minClient = NULL;

	if(!event_list.empty())
		e = event_list.top();

	// see if any of the clients have a new packet
	for(map<int, Client*>::iterator i=clients.begin(); i != clients.end(); ++i) {
		Client* c = (*i).second;
		double t = lvt + c->nextNewPacket();
		if(event_list.empty() || t < e.timestamp)
			if(minClient == NULL || minClient->nextNewPacket() > t)
				minClient = c;
	}

	// if we found a client, then schedule the Et_New at that time, and exit
	if(minClient != NULL) {
		packet* p = NULL;
		minClient->newPacket(p);
		scheduleEvent(minClient->getId(),minClient->nextNewPacket(), p);
		return true;
	}

	// incriment the lvt
	this->lvt = e.timestamp;

	// process the event based on it's type
	switch(e.type) {
		case Et_New: {
			// schedule a route event for this packet
			this->routePacket(this->id,e.p);
		} break;
		case Et_Route: {
			// schedule for either a route or delivery event
			if(e.p->to.router == this->id) {
				double deliverOffset = rngTransmit->getRandom();
				scheduleEvent(this->id, deliverOffset, e.p, Et_Deliver);
			} else {
				map<int, Router*>::iterator i;
				i = routers.find(e.p->to.router);
				if((*i).second == NULL) {
					// rebind the rngRoutePath the with correct params
					if(rngRoutePath == NULL)
						rngRoutePath = new UniformDist(0,routers.size());

					int path = rngRoutePath->getRandom();

					// select a random path
					i = routers.begin();
					for(int j=0; j < path; ++j)
						i++;
					Router* r = (*i).second;

					// route the packet
					r->routePacket(this->id, e.p);

				} else {
					Router* r = (*i).second;

					// route the packet
					r->routePacket(this->id, e.p);
				}
			}
		} break;
		case Et_Deliver: {
			Client* c = clients[e.p->to.client];

			// deliver the packet
			c->deliverPacket(e.p);
		} break;
		default: {
			cout << "Unknown event type in Router(" << this->id << ") ";
			e.dump();
			event_list.pop();
			return false;
		}
	}

//	cout << "Process event: "; e.dump();

	// remove the top event
	event_list.pop();

	return true;
}

bool Router::routePacket(int poster,  packet* p) {
	double routeOffset = rngTransmit->getRandom();

	p->hops++;

	// schedule a Et_Route event
	scheduleEvent(poster, routeOffset, p, Et_Route);

	return true;
}

bool Router::scheduleEvent(int poster, double timeOffset, packet* p, EVENT_type type) {
	Event e(poster, this->lvt + timeOffset, type, p);

//	cout << "Schedule Event: "; e.dump();

	// push the event into the event list
	event_list.push(e);

	return true;
}

int Router::getId() {
	return this->id;
}

int Router::getClientsCount() {
	return clients.size();
}

double Router::getLVT() {
	return lvt;
}

bool Router::registerAddresses(AddressRegister* ar) {
	address addr;

	addr.router = this->id;

	for(map<int, Client*>::iterator i=clients.begin(); i != clients.end(); ++i) {
		Client* c = (*i).second;
		addr.client = c->getId();
		ar->registerAddress(addr);
	}

	return true;
}

void Router::dumpStats() {
	cout << "STATS FOR ROUTER #" << this->id << endl;
	cout << "\tGENERAL STATS" << endl;
	cout << "\t\toutstanding events = " << event_list.size() << endl;

	for(map<int, Client*>::iterator i=clients.begin(); i != clients.end(); ++i) {
		Client* c = (*i).second;
		c->dumpStats();
	}
	cout << endl;
}

