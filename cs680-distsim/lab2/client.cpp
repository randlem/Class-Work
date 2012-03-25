#include "client.h"
#include "router.h"

Client::Client(int id, AddressRegister* ar) : rngSize(rngSizeBeta), rngNextNew(rngNextNewBeta) {
	this->id = id;
	this->addressRegister = ar;
	this->nextNew = rngNextNew.getRandom();
}

Client::~Client() {

}

double Client::nextNewPacket() {
	return this->nextNew;
}

bool Client::newPacket(packet*& p) {
	// if there is a value in p, we should delete it to save a memory leak
	if(p != NULL)
		delete p;

	// allocate a new packet
	p = new packet;

	// set the source address
	p->from.router = gateway->getId();
	p->from.client = this->id;
	p->to   = addressRegister->randomAddress();
	p->hops = 0;
	p->size = rngSize.getRandom();
	p->timestamp = this->gateway->getLVT() + this->nextNewPacket();

	// make the address key
	short addr = p->to.router << 8 | p->to.client;

	// record some observations
	sentMessages.observe(this->gateway->getLVT(), 1.0);
	sentSize.observe(this->gateway->getLVT(), (double)p->size);
	sendTo[addr]++;

	// get a new nextNew time
	this->nextNew = rngNextNew.getRandom();

	return true;
}

bool Client::attachRouter(Router* const r) {
	this->gateway = r;
	return true;
}

bool Client::deliverPacket(packet* p) {
	double t = this->gateway->getLVT();
	short addr = p->from.router << 8 | p->from.client;

	// record statistics
	hops.observe(t,p->hops);
	timeInTravel.observe(t,t - p->timestamp);
	recvMessages.observe(t,1.0);
	recvSize.observe(t,(double)p->size);
	recvFrom[addr]++;

	// free the memory
	delete p;
	p = NULL;

	return true;
}

int Client::getId() {
	return this->id;
}

void Client::dumpStats() {
	map<short,int>::iterator i;

	cout << "\tCLIENT #" << this->id << endl;
	cout << "\t\tGENERAL STATS" << endl;
	cout << "\t\t\tmean hops = " << this->hops.getMean() << endl;
	cout << "\t\t\tmean travel time = " << this->timeInTravel.getMean() << endl;

	cout << "\t\tSEND STATS" << endl;
	cout << "\t\t\tcount = " << this->sentMessages.getSum() << endl;
	cout << "\t\t\tmean size = " << this->sentSize.getMean() << endl;
	cout << "\t\t\tmax size = " << this->sentSize.getMaximum() << endl;
	cout << "\t\t\tmin size = " << this->sentSize.getMinimum() << endl;
	cout << "\t\t\trecipient counts by address:" << endl;
	for(i=sendTo.begin(); i != sendTo.end(); ++i)
		cout << "\t\t\t" << hex << setw(4) << setfill('0') << (*i).first << dec << " " << (*i).second << endl;

	cout << "\t\tRECV STATS" << endl;
	cout << "\t\t\tcount = " << this->recvMessages.getSum() << endl;
	cout << "\t\t\tmean size = " << this->recvSize.getMean() << endl;
	cout << "\t\t\tmax size = " << this->recvSize.getMaximum() << endl;
	cout << "\t\t\tmin size = " << this->recvSize.getMinimum() << endl;
	cout << "\t\t\tsender counts by address:" << endl;
	for(i=recvFrom.begin(); i != recvFrom.end(); ++i)
		cout << "\t\t\t" << hex << setw(4) << setfill('0') << (*i).first << dec << " " << (*i).second << endl;

}
