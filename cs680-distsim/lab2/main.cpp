#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include "router.h"

int main(int argc, char* argv[]) {
	AddressRegister ar;
	Router* r1;
	Router* r2;
	Router* r3;
	Client* r1c1;
	Client* r2c1;
	Client* r3c1;

	// create the router/client objects
	r1 = new Router(1);
	r2 = new Router(2);
	r3 = new Router(3);
	r1c1 = new Client(1,&ar);
	r2c1 = new Client(1,&ar);
	r3c1 = new Client(1,&ar);

	// create the network
	r1->attachClient(r1c1);
	r1->attachRouter(r2);

	r2->attachClient(r2c1);
	r2->attachRouter(r1);
	r2->attachRouter(r3);

	r3->attachClient(r3c1);
	r3->attachRouter(r2);

	// populate the address register so the clients know who to send messages to
	r1->registerAddresses(&ar);
	r2->registerAddresses(&ar);
	r3->registerAddresses(&ar);

	for(int i=0; i < 10000; ++i) {
		if(r1->nextEventTime() < r2->nextEventTime())
			r1->doNextEvent();
		else if(r3->nextEventTime() < r2->nextEventTime())
			r3->doNextEvent();
		else
			r2->doNextEvent();
	}

	r1->dumpStats();
	r2->dumpStats();
	r3->dumpStats();

	delete r1;
	delete r2;
	delete r1c1;
	delete r2c1;

	return(0);
}
