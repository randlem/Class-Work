/*******************************************************************************
* Lab 3 -- Scheduling Simulation
*
* WRITTEN BY: Mark Randles, Paul Jankura
*
* PURPOSE: The purpose of this file is to provide the most basic operation of
* the simulator, including any command-line parsing, object creation, running
* of the simulator, and outputting statistics from the simulator.
*
*******************************************************************************/

/*******************************************************************************
* INCLUDES
*******************************************************************************/
#include <iostream>
using std::cout;
using std::endl;

#include <string>
using std::string;

#include "SimPlus.h"
#include "processor/dummy.h"
#include "processor/fcfs.h"
#include "processor/sjn.h"
#include "processor/rr.h"
#include "processor/srtn.h"
#include "processor/hrrn.h"

/*******************************************************************************
* FUNCTIONS
*******************************************************************************/
int main(int argc, char* argv[]) {
	string type = argv[1];
	Processor* proc = NULL;

	// initlization
	cout << "CPU Scheduling Simulation" << endl;

	// create the type of processor passed
	if(type == "fcfs")
		proc = new Processor_FCFS();
//	else if(type == "sjn")
//		proc = new Processor_SJN();
	else if(type == "rr")
		proc = new Processor_RR();
	else if(type == "srtn")
		proc = new Processor_SRTN();
//	else if(type == "hrrn")
//		proc = new Processor_HRRN();
	else
		proc = new Processor_Dummy();

	// do the simulation
	proc->generate_queue();
	proc->simulate();
	proc->print_stats();

	// cleanup memory
	if(proc != NULL) {
		delete proc;
		proc = NULL;
	}

	// exit
	cout << "Finished." << endl;
	return(0);
}
