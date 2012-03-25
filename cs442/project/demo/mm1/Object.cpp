#include "Object.h"

SampST* Object::totalTimeStat = SimPlus::getInstance()->getSampST();

Object::Object() {
	entryTime = 0;
	startedWaiting = 0;
}

Object::~Object() {

}

void Object::beginWait( double simTime ) {
	startedWaiting = simTime;
}

double Object::endWait( double simTime ) {
	return (simTime - startedWaiting);
}

void Object::enterSystem( double simTime ) {
	entryTime = simTime;
}

void Object::exitSystem( double simTime ) {
	totalTimeStat->observe( simTime, simTime - entryTime );
}

void Object::getStats(ostream& out) {
	out << "Average Object's total turnaround time: "
		<< Object::totalTimeStat->getMean() << endl << endl;
}
