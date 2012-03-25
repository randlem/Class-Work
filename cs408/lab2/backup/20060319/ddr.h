#ifndef __DDR_H__
#define __DDR_H__

/**********************************************************
* INCLUDES
**********************************************************/
#include <queue>
using std::queue;

#include <vector>
using std::vector;

#include <map>
using std::map;

#include <string>
using std::string;

#include <math.h>
#include <stdlib.h>
#include <unistd.h>

/**********************************************************
* MACROS
**********************************************************/
#define RANDOM() ((double)(rand() / (double)RAND_MAX))
#define DEBUG_PRINT(a) if(debug) { cout << a; }
#define SLEEP(a) usleep((useconds_t)(double)(a * (double)1000))

/**********************************************************
* CONSTANTS
**********************************************************/
const int	N = 8;				// number of processes
const float	S = 120.0;			// request time
const int   T = 256;			// number of tracks
const float	M = 0.05;			// track seek time
const float V = 3.0;			// overhead time per request
const int	REQUESTS = 10;	// number of requests to run the sim for

/**********************************************************
* GLOBALS
**********************************************************/
bool			debug = false; 			// set to true then debug info will print

/**********************************************************
* STRUCTURES
**********************************************************/

// a structure to hold information about a request to the disk drive
typedef struct request {
	int		thread;			// thread id of the requesting thread
	int		track;			// track # of the request
	double	time_offset;	// time offset from the last request
};

// abstract base class for request queues
class RequestQueue {
public:
	// base constructor
	RequestQueue() {
		total = 0;
	}

	// base destructor
	virtual ~RequestQueue() {

	};

	// returns the next requests that should be processed
	virtual request* next_request() = 0;

	// returns the current # of requests pending
	virtual int request_count() = 0;

	// returns the total requests processed;
	int request_total() {
		return(total);
	}

	// adds a new request to the queue
	virtual bool new_request(request *r) = 0;

	// performs a dump of the current queue state
	virtual void queue_dump() = 0;

	virtual void print_stats() = 0;

protected:
	int total;
	string name;
};

// a request queue which implements a first-in, first-out priority
class RequestQueueFIFO : public RequestQueue {
public:
	RequestQueueFIFO() {
		name = "FIFO";
	}

	virtual ~RequestQueueFIFO() {
		// clean up any requests left in the queue
		while(!fifo.empty()) {
			delete fifo.front();
			fifo.pop();
		}
	}

	request* next_request() {
		request *r;	// temp request pointer

		// get the top request and pop the value off the queue
		r = fifo.front();
		fifo.pop();

		// return the value of the pointer
		return(r);
	}

	int request_count() {
		// return the count
		return(fifo.size());
	}

	bool new_request(request* r) {
		// make sure we didn't get a null object
		if(r == NULL)
			return(false);

		// push the new request into the queue
		fifo.push(r);

		// incriment the total counter
		total++;

		// well everything work so return something that represents that
		return(true);
	}

	void queue_dump() {
		queue<request*> t = this->fifo;
		request* r = NULL;
		int i = 0;
		cout << "BEGIN QUEUE DUMP -------------------------" << endl;
		while(!t.empty()) {
			r = t.front();
			t.pop();

			cout << "#" << i << ": request = (" << r->thread << ", " << r->track << ", " << r->time_offset << ")" << endl;
			i++;
		}
		cout << "END QUEUE DUMP ---------------------------" << endl;
	}

	void print_stats() {
		cout << "Queue Type: " << name << endl;
		cout << "Total Requests: " << total << endl;
	}

private:
	queue<request*> fifo;	// STL queue class which is a FIFO queue

};

// A request queue which implements the SCAN priority algorithm
class RequestQueueSCAN : public RequestQueue {
public:
	RequestQueueSCAN() {
		head_position = 0;
		direction = 1;
		name = "SCAN";
	}

	virtual ~RequestQueueSCAN() {
		// clean up any requests left in the queue
		for(int i=0; i < T; i++)
			for(vector<request*>::iterator j=requests[i].begin(); j != requests[i].end(); ++j)
				delete *j;
	}

	request* next_request() {
		request *r;	// temp request pointer

		while(requests[head_position].size() <= 0) {
			head_position += direction;

			if(head_position < 0) {
				head_position = 0;
				direction *= -1;
			}

			if(head_position >= T) {
				head_position = T - 1;
				direction *= -1;
			}
		}

		// get the first element in the
		r = requests[head_position].front();
		requests[head_position].erase(requests[head_position].begin());

		// decriment the counter
		count--;

		// return the value of the pointer
		return(r);
	}

	int request_count() {
		// return the count
		return(count);
	}

	bool new_request(request* r) {
		// make sure we didn't get a null object
		if(r == NULL)
			return(false);

		// insert the request into the vector for that track
		requests[r->track].push_back(r);

		// incriment the total counter
		total++;
		count++;

		DEBUG_PRINT(requests[r->track].size() << endl);

		// well everything work so return something that represents that
		return(true);
	}

	void queue_dump() {

	}

	void print_stats() {
		cout << "Queue Type: " << name << endl;
		cout << "Total Requests: " << total << endl;
	}

private:
	map<int, vector<request*> > requests;
	int head_position;
	int direction;
	int count;
};

#endif
