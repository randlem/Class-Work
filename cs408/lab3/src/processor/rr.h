/*******************************************************************************
* rr.h -- Definition and Implementation of the Round-Robin Scheduler Simulator
*
* WRITTEN BY: Mark Randles and Paul Jankura
*
* PURPOSE: The purpose of this class is to simulate as closely as possible a
* Round-Robin type scheduler and gather statistics about it's operation.
*
*******************************************************************************/

#ifndef __RR_H__
#define __RR_H__

#include "processor.h"

const int DEFAULT_TIME_SLICE = 10.0;

class Processor_RR : public Processor {
public:
	Processor_RR(int time_slice = DEFAULT_TIME_SLICE) {
		processor_name = "Round Robin";
		this->time_slice = time_slice;
	}

	void simulate() {
		Process_ptr p = NULL;
		queue<Process_ptr> ready_queue;

		do {
			// if the ready queue is empty, push the first event on it and update
			// the global time
			if(ready_queue.empty()) {
				p = arrival_queue.front();
				ready_queue.push(p);
				arrival_queue.pop();
				time = p->time_arrival;
				time_unused.observe(time,time);
			}

			// get the process for the next RR slice
			p = ready_queue.front();
			ready_queue.pop();

			// calc the ending of this time slice and either push a unfinished
			// process back in the ready queue or remove it
			if(p->time_left >= time_slice) {

				// set the time this process has left
				p->time_left -= time_slice;

				// incriment the waiting time
				p->time_wait += time - p->time_last;

				// incriment the cpu time and cpu burst counter
				p->time_cpu += time_slice;
				p->burst_cpu++;

				// check to see if this is the first time the process has been worked on
				if(p->time_last <= 0)
					time_response.observe(time,time - p->time_arrival);

				// set the new time
				time += time_slice;

				// assign the last time this process was operated on
				p->time_last = time;

				// push the current process on the back of the ready queue
				ready_queue.push(p);

			} else {
				// incriment for the last bit of cpu time
				p->time_cpu += p->time_left;
				p->burst_cpu++;

				// incriment the waiting time
				p->time_wait += time - p->time_last;

				// set the new time
				time += p->time_left;

				// gather some statistics
				time_process.observe(time, p->time_cpu);
				time_wait.observe(time, p->time_wait);
				time_turnaround.observe(time, time - p->time_arrival);
				finished++;

				// clean up my memory
				delete p;
			}

			// get any processes out of the arrival queue that are going to arrive
			// during the duration of this RR slice
			while(!arrival_queue.empty() && !ready_queue.empty() && (time > arrival_queue.front()->time_arrival)) {
				ready_queue.push(arrival_queue.front());
				arrival_queue.pop();
			}

		} while(!ready_queue.empty());
	}

private:
	int time_slice;
};

#endif
