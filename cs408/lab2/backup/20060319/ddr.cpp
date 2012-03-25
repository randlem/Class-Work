/************************************************
* lab2.cpp
*
* Name:  Mark Randles
* Class: CS408
* Date:  2006-02-28
*
* Description: Real-time simulation of disk
*  access, based on a total service time for
*  1000 disk access requests.
************************************************/

/**********************************************************
* INCLUDES
**********************************************************/
#include <iostream>
using std::cout;
using std::endl;

#include <map>
using std::map;

#include <unistd.h>
#include <pthread.h>

#include "ddr.h"

/**********************************************************
* GLOBALS
**********************************************************/
bool			done = false;			// set to true if the disk drive has completed REQUESTS requests
RequestQueue	*request_queue = NULL;	// the request queue for the disk drive
pthread_mutex_t mutex_request_queue = PTHREAD_MUTEX_INITIALIZER;
bool 			complete[N];			// set to true when the disk_drive has completed the last request

/**********************************************************
* PROTOTYPES
**********************************************************/
void *disk_drive();
void *request_thread(void*);

/**********************************************************
* MAIN FUNCTION
**********************************************************/
int main(int argc, char* argv[]) {
	pthread_t threads[N];	// index of threads

	request_queue = NULL;

	cout << argc << endl;

	if(argc > 1 && (strcmp(argv[1],"--debug") == 0 || strcmp(argv[1],"-d") == 0)) {
		debug = true;
		DEBUG_PRINT("Debug ON" << endl)
		if(argc >= 2 && (strcmp(argv[2],"--scan") == 0))
			request_queue = new RequestQueueSCAN();
		else
			request_queue = new RequestQueueFIFO();
	} else if(argc > 1 && (strcmp(argv[1],"--scan") == 0))
		request_queue = new RequestQueueSCAN();
	else
		request_queue = new RequestQueueFIFO();

	// make sure a request queue exists
	if(request_queue == NULL) {
		return(1);
	}

	// seed the random # generator
	srand(0);

	// create the request threads
	for(int i=0; i < N; i++)
		pthread_create(&threads[i], NULL, request_thread, (void *)i);

	// run the disk_drive() process, so we've only got N+1 threads
	disk_drive();

	// print some stats for the queue
	request_queue->print_stats();

	return(0);
}

/**********************************************************
* FUNCTIONS
**********************************************************/
void *disk_drive() {
	double total_time = 0.0; // sum of all the times
	int served = 0; // the total number of request processed, should = REQUESTS
	double seek_time = 0.0; // the seek time for the request
	request *r = NULL;

	// do a service loop until we've processed N requests
	while(served < REQUESTS) {
		// see if there's a waiting request
		while(request_queue->request_count() <= 0) {
			usleep(1); // sleep the thread for a bit if there is no request
		}

		// get the next request
		pthread_mutex_lock(&mutex_request_queue);
		r = request_queue->next_request();
		DEBUG_PRINT("Service #: " << served << " (" << r->thread << "," << r->track << "," << r->time_offset << ")" << endl);
//		request_queue->queue_dump();
		pthread_mutex_unlock(&mutex_request_queue);

		// get the seek time
		seek_time = V + (r->track * M);

		// add the current seek time to the total service time
		total_time += seek_time;

		// since we've successfully served one request, inc our counter
		served++;

		// sleep for the seek time
		SLEEP(seek_time);

		// set the request completed flag
		complete[r->thread] = true;
	}

	// set the done flag to signal the request processes to terminate
	done = true;

	// unblock all of the request threads that are still blocked
	memset(&complete,true,sizeof(bool) * N);

	// print out the average service time
	cout << "Average Service Time = " << (float)(total_time / served) << endl;
//	request_queue->queue_dump();

	// exit the thread
	return(0);
//	pthread_exit(NULL);
}

void *request_thread(void *thread_id) {
	double delay = 0.0;
	double total_delay = 0.0;
	int total_requests = 0;
	request *r = NULL;

	while(!done) {
		// create a new request object
		r = NULL;
		r = new request;

		// setup the new request
		delay = (double)(RANDOM() * S);
		r->track = (int)(RANDOM() * T);
		r->time_offset = delay;
		r->thread = (int)thread_id;
		complete[(int)thread_id] = false;

		// add the delay time to the total delay time for this thread
		total_delay += delay;

		// sleep until the request needs to be posted
		SLEEP(delay);

		// insert the new request into the queue
		pthread_mutex_lock(&mutex_request_queue);
		request_queue->new_request(r);
		DEBUG_PRINT("Thread " << (int)thread_id << " created new request (" << r->track << "," << delay << ")" << endl);
//		request_queue->queue_dump();
		pthread_mutex_unlock(&mutex_request_queue);

		// wait for the request to complete
		while(!complete[(int)thread_id]) { }

		// incriment the total requests count for this thread
		total_requests++;

	}

	cout << "Thread id: " << (int)thread_id << "; Average Request Time: " << (double)(total_delay / (double)total_requests) << "; Total Requests Made: " << total_requests << endl;

	// exit the thread
	pthread_exit(NULL);
}
