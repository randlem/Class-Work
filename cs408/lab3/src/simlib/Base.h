#include "LogFile.h"

#ifndef BASE_H
#define BASE_H

enum logLEVEL { ERROR,  WARNING, ALL };

class Base {
		Base(logLEVEL);
		~Base();

		void setLogLevel(logLEVEL);

		void writeError(string);
		void writeWarning(string);
		void writeMessage(string);

		string formatInt(int);
		string formatDouble(double);

	private:
		static LogFile log;
		static LogLEVEL level;
		static bool started;
};

#endif