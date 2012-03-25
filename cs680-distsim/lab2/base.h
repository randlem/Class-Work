/*******************************************************************************
* base.h -- Basic data structures and types
*
* PURPOSE: To aggragate all of the small enums, unions, strucs, and classes
* into one file that is easier to include then many small/large ones.
*
*******************************************************************************/

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::hex;
using std::dec;

#ifndef BASE_H
#define BASE_H

enum EVENT_type {
	Et_New,
	Et_Route,
	Et_Deliver
};

typedef struct {
	char client;
	char router;
} address;

typedef struct {
	address from;
	address to;
	int hops;
	double timestamp;
	int size;
} packet;

class Event {
public:
	Event() { ; }

	Event(int poster, double timestamp, EVENT_type type, packet* p) {
		this->poster    = poster;
		this->timestamp = timestamp;
		this->type 		= type;
		this->p			= p;
	}

	int			poster;
	double 		timestamp;
	EVENT_type 	type;
	packet* 	p;

	bool operator > (const Event& e) const {
		return(this->timestamp > e.timestamp);
	}

	void dump() {
		cout << this->timestamp << " ";
		switch(this->type) {
			case Et_New: {
				cout << "Et_New";
			} break;
			case Et_Route: {
				cout << "Et_Route";
			} break;
			case Et_Deliver: {
				cout << "Et_Deliver";
			} break;
			default: {
				cout << (int)this->type;
			} break;
		}
		cout << " " << hex << this->p << dec << endl;
	}

private:


};

#endif
