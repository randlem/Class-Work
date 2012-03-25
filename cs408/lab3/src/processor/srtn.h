/*******************************************************************************
* srtn.h -- Definition and Implementation of the Shortest Remaining Time Next
*     Scheduler Simulator
*
* WRITTEN BY: Mark Randles and Paul Jankura
*
* PURPOSE: The purpose of this class is to implement as closely as possible
* a simulation of the Shortest Remaining Time Next scheduling algorithm and
* collect relevant statistics about it.
*
*******************************************************************************/
#ifndef __SRTN_H__
#define __SRTN_H__

#include <vector>
using std::vector;

#include "processor.h"

class Processor_SRTN : public Processor {
public:
	Processor_SRTN() {
		processor_name = "Shortest Remaining Time Next";
	}

	void simulate() {
		Process_ptr p = NULL;

		// do a recursion until we run out of events
		while(!arrival_queue.empty()) {
			// set a flag to run some code saying we went out of the recursion
			is_busy = false;

			// seed the recursion with a good process
			p = arrival_queue.front();
			arrival_queue.pop();

			// do the recursion
			schedule(p);
		}
	}

	void schedule(Process_ptr p) {
		Process_ptr next = NULL;

		// if the cpu is not busy, the we should record the time it spend in
		// an unused state
		if(!is_busy) {
			if(time < p->time_arrival) {
				time_unused.observe(time,p->time_arrival - time);
				time = p->time_arrival;
			}
			is_busy = true;
		}

		// record any time this process waited to get the CPU for the first time
		// or again
		p->time_wait += time - p->time_arrival;

		// record the response time of this process
		time_response.observe(p->id,time - p->time_arrival);

		while(p->time_left > 0) {
			next = arrival_queue.front();
			if(next != NULL && p->time_left > next->time_left &&
				next->time_arrival < time + p->time_left) {
				// update the current process to the correct stats at the time
				// of preemption
				p->time_left -= next->time_arrival - time;
				p->time_cpu += next->time_arrival - time;
				p->burst_cpu++;

				// update the global time to the time of the preemption
				time += next->time_arrival - time;
				p->time_last = time;

				// do a recurions (preemption)
				arrival_queue.pop();
				schedule(next);

				// calc the wait time for this process
				p->time_wait += time - p->time_last;
			} else {
				// otherwise we are not going to preempt before this process
				// finishes so we're going to update it to it's finished stats
				// and break the loop
				time += p->time_left;
				p->time_cpu += p->time_left;
				p->time_left -= p->time_left;
				p->burst_cpu++;
				break;
			}
		}

		// make some observations
		time_process.observe(p->id,p->time_cpu);
		time_wait.observe(p->id,p->time_wait);
		time_turnaround.observe(p->id,time - p->time_arrival);
		finished++;

		return;
	}

private:
	bool is_busy;	// flag to indicate if there is a job on the processor, needed to find any unused processor time
};

#endif
