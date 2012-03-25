/*******************************************************************************
* dummy.h -- Definition and Implementation of a Dummy Simulator Object
*
* WRITTEN BY: Mark Randles and Paul Jankura
*
* PURPOSE: The purpose of this class is to provide a basic processor object
* which does some useful thing, namely dumping the arrival queue to std out.
*
*******************************************************************************/
#ifndef __DUMMY_H__
#define __DUMMY_H__

#include "processor.h"

class Processor_Dummy : public Processor {
public:
	Processor_Dummy() {
		processor_name = "Dummy Processor";
	}

	void simulate() {
		Process_ptr p = NULL;

		// dump the arrival queue
		while(!arrival_queue.empty()) {
			p = arrival_queue.front();
			p->print();
			arrival_queue.pop();
		}
	}

private:

};

#endif
