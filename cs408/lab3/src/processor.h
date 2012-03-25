/*******************************************************************************
* Processor.h
*
* DESCRIPTION: Describes a virtual base class for different processor
* schedulers.
*******************************************************************************/

#ifndef __PROCESSOR_H__
#define __PROCESSOR_H__

/*******************************************************************************
* INCLUDES
*******************************************************************************/
#include <iostream>
using std::cout;
using std::endl;

#include <queue>
using std::queue;

#include <string>
using std::string;

#include "SimPlus.h"

/*******************************************************************************
* CONSTANTS
*******************************************************************************/
const int DEFAULT_PROCESSES = 50;
const double DEFAULT_ARRIVAL_THETA = 10.0;
const double DEFAULT_TOTAL_THETA = 100.0;
const double TIME_SCALAR = 1;

/*******************************************************************************
* STRUCTS
*******************************************************************************/
class Process{
public:
	void print() {
		int t = cout.precision();
		cout.precision(10);
		cout << "Process ID #"			<< id << endl;
		cout << "\tArrival Time\t= " 	<< time_arrival << endl;
		cout << "\tCPU Time Left\t= "	<< time_left << endl;
		cout << "\tWaiting Time\t= "	<< time_wait << endl;
		cout << "\tCPU Time\t= "		<< time_cpu << endl;
		cout << "\tCPU Bursts\t= "		<< burst_cpu << endl;
		cout << endl;
		cout.precision(t);
	}

	int id;					// process ID
	double time_arrival;	// time the processes arrives in the queue
	double time_left;		// time left to complete the process' task
	double time_wait;		// total time the process spends waiting in the queue
	double time_cpu;		// total time the process spends working on the CPU
	double time_last;		// the time it was last worked on
	int burst_cpu;			// number of CPU bursts allocated to the process
};

typedef Process* Process_ptr;

/*******************************************************************************
* CLASSES
*******************************************************************************/
class Processor {
public:
	Processor(int processes = DEFAULT_PROCESSES,
		double arrival_theta = DEFAULT_ARRIVAL_THETA,
		double total_theta = DEFAULT_TOTAL_THETA) {
		this->time = 0.0;
		this->finished = 0;
		this->processes = processes;
		this->arrival_theta = arrival_theta;
		this->total_theta = total_theta;
	}

	virtual ~Processor() {
		// cleanup any memory that may have gotten stranded in the queue
		while(!arrival_queue.empty()) {
			delete arrival_queue.front();
			arrival_queue.pop();
		}
	}

	int generate_queue() {
		ExponentialDist rng_arrival(arrival_theta,RNGFactory::Local);
		ExponentialDist rng_total(total_theta,RNGFactory::Local);
		double t = 0.0;
		Process_ptr p = NULL;

		cout << "Process List (Order of Creation)" << endl;
		for(int i=0; i < processes; ++i) {
			// allocate the memory for a new process
			p = new Process();

			// zero out the struct
			memset(p,0,sizeof(Process));

			// set the values that need set
			p->id = i;
			t = p->time_arrival = TIME_SCALAR * rng_arrival.getRandom() + t;
			p->time_left = TIME_SCALAR * rng_total.getRandom();

			// print the newly created process
			p->print();

			// insert the new process into the queue
			arrival_queue.push(p);

			// record some stats about it
			time_arrival.observe(p->id,p->time_arrival);
			time_needed.observe(p->id,p->time_left);
		}
		cout << "END LIST" << endl;

		// return the number of processes created
		return(arrival_queue.size());
	}

	virtual void simulate() = 0;

	void print_stats() {
		// dump the statistics
		int t = cout.precision();
		cout.precision(10);
		cout << "Statistics for Processor " << processor_name << endl;
		cout << endl;
		cout << "\tTURNAROUND TIME: " << endl;
		cout << "\t\tMin\t = " << time_turnaround.getMinimum() << endl;
		cout << "\t\tMax\t = " << time_turnaround.getMaximum() << endl;
		cout << "\t\tAvg\t = " << time_turnaround.getMean() << endl;
		cout << "\t\tSize\t = " << time_turnaround.getSampleSize() << endl;
		cout << endl;
		cout << "\tWAITING TIME: " << endl;
		cout << "\t\tMin\t = " << time_wait.getMinimum() << endl;
		cout << "\t\tMax\t = " << time_wait.getMaximum() << endl;
		cout << "\t\tAvg\t = " << time_wait.getMean() << endl;
		cout << "\t\tSize\t = " << time_wait.getSampleSize() << endl;
		cout << endl;
		cout << "\tRESPONSE TIME: " << endl;
		cout << "\t\tMin\t = " << time_response.getMinimum() << endl;
		cout << "\t\tMax\t = " << time_response.getMaximum() << endl;
		cout << "\t\tAvg\t = " << time_response.getMean() << endl;
		cout << "\t\tSize\t = " << time_response.getSampleSize() << endl;
		cout << endl;
		cout << "\tCPU TIME: " << endl;
		cout << "\t\tMin\t = " << time_process.getMinimum() << endl;
		cout << "\t\tMax\t = " << time_process.getMaximum() << endl;
		cout << "\t\tAvg\t = " << time_process.getMean() << endl;
		cout << "\t\tSize\t = " << time_process.getSampleSize() << endl;
		cout << endl;
		cout << "\tARRIVAL TIME: " << endl;
		cout << "\t\tMin\t = " << time_arrival.getMinimum() << endl;
		cout << "\t\tMax\t = " << time_arrival.getMaximum() << endl;
		cout << "\t\tAvg\t = " << time_arrival.getMean() << endl;
		cout << "\t\tSize\t = " << time_arrival.getSampleSize() << endl;
		cout << endl;
		cout << "\tNEEDED TIME: " << endl;
		cout << "\t\tMin\t = " << time_needed.getMinimum() << endl;
		cout << "\t\tMax\t = " << time_needed.getMaximum() << endl;
		cout << "\t\tAvg\t = " << time_needed.getMean() << endl;
		cout << "\t\tSize\t = " << time_needed.getSampleSize() << endl;
		cout << endl;
		cout << "\tUNUSED TIME: " << endl;
		cout << "\t\tMin\t = " << time_unused.getMinimum() << endl;
		cout << "\t\tMax\t = " << time_unused.getMaximum() << endl;
		cout << "\t\tAvg\t = " << time_unused.getMean() << endl;
		cout << "\t\tSize\t = " << time_unused.getSampleSize() << endl;
		cout << endl;
		cout << "\tGENERAL: " << endl;
		cout << "\t\tThroughput\t\t = " << (finished / time) << endl;
		cout << "\t\tUtilitization (in %)\t = " << (1 - (time_unused.getSum() / time_process.getSum())) * 100 << endl;
		cout.precision(t);
	};

protected:
	// options
	int processes;			// the number of processes that should be generated
	double arrival_theta;	// the theta of an exponential dist to determine the arrival time
	double total_theta;		// the theta of an exponential dist to determine the total time a new process will take to complete
	string processor_name; 	// algorithm name for output

	// process queue
	queue<Process_ptr> arrival_queue;	// a queue which contains all of the processes in order of arrival

	// clock
	double time;	// the current time in the simulation

	// stat collectors
	int finished;			// a count of the number of threads that have finished
	SampST time_process;	// a stat to contain the actual time the processor is busy
	SampST time_wait;		// a stat to contain the total wait time for processes
	SampST time_response;	// a stat to contain the time spent in the queue from arrival to first execution
	SampST time_turnaround; // a stat to contain the total turnaround time for processes
	SampST time_unused;		// a stat to contain the total amount of time the processor spends not doing anything
	SampST time_arrival;	// a stat to contain the arrival times of the processes
	SampST time_needed;		// a stat to contain the needed time of each process
};

#endif
