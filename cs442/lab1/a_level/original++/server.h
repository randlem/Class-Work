#include <queue>
using std::queue;

#ifndef SERVER_H
#define SERVER_H

class Server {
public:
	Server();
	~Server();

	bool status() const;   // get the server status
	double getNextTime() const;  // returns the event time of the next event associated with this server
	double doNextEvent();  // perform the next event returning the time that event occurs

private:
	queue<double> timeArrival;

}


#endif
