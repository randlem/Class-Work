#include <string>
using std::string;

#ifndef EXCEPTION_H
#define EXCEPTION_H

class Exception {
public:
	Exception(char* c) : error(c) { ; }
	Exception(string &s) : error(s) { ; }
	Exception() : error("General Error") { ; }

	string error;

};

#endif
