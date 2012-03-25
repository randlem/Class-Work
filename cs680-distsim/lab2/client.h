/*******************************************************************************
* client.h -- Definition of the Client object
*
* PURPOSE: This class is designed to act a both a source and sink of packets.
* Each client is unable to connect to any other client, and can only communicate
* by using the Router to which it's attached.  It's essentially a collection of
* distributions and stat collectors that represents a Client on the network.
*
* IMPLEMENTATION: The Client is the probability model for a client in the
* simulation.  Each Client contains the needed distributions to tell the
* Router object that it's attached to when this Client's next message should
* be scheduled and where it's being sent to.  Before a Router processes it's
* events, it should call the nextNewPacket() method of every attached Client
* to see if there are any new packets that will occur before the already
* scheduled event.  If there is, then the Router should call newPacket() and
* get a pointer to the new packet.  It's then the Router objects responsibility
* to send this packet on to the destination.
*
* The method deliverPacket() takes a pointer ot a packet and draws what
* statistical information needed from it and then discards it.  It may be
* desirable for the packets to be kept to modify the distributions that effect
* when and where a new packet will go.  To more realistically simulate a
* network, a delivered packet would almost always trigger an new packet to be
* sent to the same source.  For simplicity sake this is being ignored.
*
*******************************************************************************/

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::hex;
using std::dec;

#include <iomanip>
using std::setw;
using std::setfill;

#include "base.h"
#include "address.h"
#include "simplus/SampST.h"
#include "simplus/ExponentialDist.h"

#ifndef CLIENT_H
#define CLIENT_H

class Router;

const double rngSizeBeta = 10.0;
const double rngNextNewBeta  = 1.0;

class Client {
public:
	Client(int, AddressRegister*);
	~Client();

	double nextNewPacket();
	bool newPacket(packet*&);
	bool deliverPacket(packet*);

	bool attachRouter(Router* const);

	int getId();
	void dumpStats();

private:
	int id;
	double nextNew;
	AddressRegister* 	addressRegister;
	Router* 			gateway;

	ExponentialDist 	rngSize;
	ExponentialDist		rngNextNew;

	SampST 				hops;
	SampST 				timeInTravel;
	SampST				sentMessages;
	SampST				sentSize;
	SampST				recvMessages;
	SampST				recvSize;
	map<short,int>		sendTo;
	map<short,int>		recvFrom;
};

#endif
