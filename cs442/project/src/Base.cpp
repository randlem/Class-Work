#include "Base.h"

LogFile Base::log("simLog.txt");
logLEVEL Base::level;

Base::Base() {
	level = ALL;
	if(!started)
		started = true;
}

Base::~Base() {
	if(started) {
		log.stop();
		started = false;
	}
}

void setLogLevel(logLEVEL l) {
	level = l;
}

void Base::writeError(string) {

}

void Base::writeWarning(string) {

}

void Base::writeMessage(string) {

}

string Base::formatInt(int) {

}

string Base::formatDouble(double) {

}
