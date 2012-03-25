/****************************************************************************

LogFile.h

Originally Written by: Mark Randles and Dan Sinclair

Asynchronos output of data formatted as a STL string.

METHODS:
--------

LogFile()
Empty constructor, set's all parameters to defaults.

LogFile(string)
Constructor that takes a string as a filename.

LogFile(string,int)
Constructor that takes a string as a filename and a integer as a buffer size.

LogFile(string,int,int)
Constructor that takes a string as a filename, a integer as a buffer size, and
a second integer as the iostream flush rate in characters.

~LogFile()
Destructor.  Closes the file and destroys the mutex.

writeString(string&) : void
Copies the string provided into the write buffer. Prolly needs optimized.

operator <<(string&) : LogFile&
Operator wrapper over writeString().

start() : void
Starts the pthread to do the async. write seperate from the calling thread.

stop() : void
Stops the pthread doing the write operations.

****************************************************************************/

#include <fstream>
using std::ofstream;
using std::endl;

#include <iostream>
using std::cout;

#include <string>
using std::string;

#include <pthread.h>

#include "BoundedBuffer.h"

#ifndef LOGFILE_H
#define LOGFILE_H

const string DEFAULT_FILENAME    = "log.txt";
const int    DEFAULT_BUFFER_SIZE = 1024;
const int    DEFAULT_FLUSH_RATE  = 1024;

void *startThread(void*);

class LogFile {
public:
	LogFile();
	LogFile(string);
	LogFile(string,int);
	LogFile(string,int,int);

	~LogFile();

	void writeString(string&);
	LogFile& operator <<(string&);

	void start();
	void stop();

private:
	pthread_t threadID;
	pthread_mutex_t mutex;

	bool isRunning;
	bool isDone;
	string fileName;
	ofstream fileStream;
	BoundedBuffer<char> writeBuffer;
	int flushRate;
};

#endif
