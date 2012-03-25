/****************************************************************************

LogFile.cpp

Originally Written by: Mark Randles and Dan Sinclair

Asynchronos output of data formatted as a STL string.  For more info see LogFile.h

****************************************************************************/

#include "LogFile.h"

LogFile::LogFile() : writeBuffer(DEFAULT_BUFFER_SIZE), isRunning(false), isDone(false),
                     fileName(DEFAULT_FILENAME), flushRate(DEFAULT_FLUSH_RATE) {
	// create the pthread
	pthread_create(&threadID,NULL,startThread,this);

	// create the mutex
	pthread_mutex_init(&mutex,NULL);
}

LogFile::LogFile(string fN) : writeBuffer(DEFAULT_BUFFER_SIZE), isRunning(false), isDone(false),
                     fileName(fN), flushRate(DEFAULT_FLUSH_RATE) {
	// create the pthread
	pthread_create(&threadID,NULL,startThread,this);

	// create the mutex
	pthread_mutex_init(&mutex,NULL);}

LogFile::LogFile(string fN, int bS) : writeBuffer(bS), isRunning(false), isDone(false),
                     fileName(fN), flushRate(DEFAULT_FLUSH_RATE) {
	// create the pthread
	pthread_create(&threadID,NULL,startThread,this);

	// create the mutex
	pthread_mutex_init(&mutex,NULL);}

LogFile::LogFile(string fN, int bS, int fR) : writeBuffer(bS), isRunning(false), isDone(false),
                     fileName(fN), flushRate(fR) {
	// create the pthread
	pthread_create(&threadID,NULL,startThread,this);

	// create the mutex
	pthread_mutex_init(&mutex,NULL);
}

LogFile::~LogFile() {
	// close the file stream
	fileStream.close();

	// destroy the mutex
	pthread_mutex_destroy(&mutex);
}

void LogFile::writeString(string& s) {
	int i;

	// copy the passed string into the writeBuffer character-by-character
	for(i=0; i < s.size(); ++i) {
		pthread_mutex_lock(&mutex);
		writeBuffer.put(s[i]);
		pthread_mutex_unlock(&mutex);
	}
}

LogFile& LogFile::operator <<(string& s) {
	// call writeString
	writeString(s);
}

void LogFile::start() {
	char c;
	int cnt = 0;

	// set the running flag to a valid state
	isRunning = true;

	// open the output stream
	fileStream.open(fileName.c_str());

	// while the running flag is true, get a character and write it to the buffer
	while(isRunning) {
		writeBuffer.get(c);
		fileStream.put(c);

		// when the count passes our flushRate we flush the stream and reset the count
		if(++cnt >= flushRate) {
			fileStream.flush();
			cnt = 0;
		}
	}

	// set the isDone flag
	isDone = true;
}

void LogFile::stop() {
	char c;

	// make the write thread stop
	isRunning = false;
	isDone = false;

	// wait for it to stop
	while(!isDone) { ; }
	pthread_join(threadID,NULL);

	// make sure the write buffer get's emptied
	while(!writeBuffer.empty()) {
		writeBuffer.get(c);
		fileStream.put(c);
	}

	// flush the output stream
	fileStream.flush();

	// tear down the writeBuffer
	writeBuffer.tearDown();
}

// needed to create a pthread, pass ptr is a pointer (this) to the calling class
void * startThread(void * ptr) {
	LogFile* lg = (LogFile*)ptr;
	lg->start();
}
