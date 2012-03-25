#ifndef __UTIL_H__
#define __UTIL_H__

#ifndef DEBUG
#define DEBUG 0
#endif

#include <iostream>
using std::cerr;
using std::endl;

#include <string>
using std::string;

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

bool handleError(string message) {
	cerr << message << endl;
	return false;
}

void debug(const char * s, ...) {
	if (DEBUG > 0) {
		va_list args;
		va_start(args, s);
		vfprintf(stdout, s, args);
		fprintf(stdout, "\n");
		va_end(args);
	}
}

#endif
