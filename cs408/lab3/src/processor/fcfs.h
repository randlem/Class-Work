/*******************************************************************************
* fcfs.h -- Definition and Implementation of the First Come, First Serve
*     Scheduler Simulator
*
* WRITTEN BY: Mark Randles and Paul Jankura
*
* PURPOSE: The purpose of this class is to simulate as closely as possible a
* First Come, First Serve type scheduling algorithm and collect relevant
* statistics about it.
*
*******************************************************************************/
#ifndef __FCFS_H__
#define __FCFS_H__

#include "processor.h"

class Processor_FCFS : public Processor {
public:
	Processor_FCFS() {
		processor_name = "First Come, First Serve";
	}

	void simulate() {
		Process_ptr p = NULL;
		double next_time = 0.0;

		// while there is a process in the arrival queue
		while(!arrival_queue.empty()) {
			// get the next process
			p = arrival_queue.front();

			// see if the process was blocked and record the time otherwise
			// record a stat marking it otherwise
			if(time > p->time_arrival)
				p->time_wait += time - p->time_arrival;
			else {
				time_unused.observe(time,p->time_arrival - time);
				time = p->time_arrival;
			}

			// record some statistics about the processor
			time_process.observe(p->id, p->time_left);
			time_wait.observe(p->id, p->time_wait);
			time_response.observe(p->id, p->time_wait);
			time_turnaround.observe(p->id, p->time_left);
			finished++;

			// set the current time to whatever the time is at the end of the process
			time += p->time_left;

			// pop the finished process off the stack
			arrival_queue.pop();

			// clear up memory
			delete p;
		}
	}

private:

};

#endif
