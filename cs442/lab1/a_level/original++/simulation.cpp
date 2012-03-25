#include <iostream>
using std::cout;
using std::endl;

#include <queue>
using std::queue;

#include "simulation.h"
#include "lcgrand.h"

Simulation::Simulation(int totalCustomersDelayed, float meanInterarrivalTime, float meanServiceTime) {
	this->meanInterarrivalTime  = meanInterarrivalTime;
	this->meanServiceTime       = meanServiceTime;
	this->totalCustomersDelayed = totalCustomersDelayed;
    simulationTime              = 0.0;
	serverState                 = ssIdle;
    timeLastEvent               = 0.0;
    numberCustomersDelayed      = 0;
    totalTimeCustomersDelayed   = 0.0;
    areaNumberInQueue           = 0.0;
    areaServerStatus            = 0.0;
	nextEventType               = etDummy;
    timeNextEvent[etArrival]    = simulationTime + expon(this->meanInterarrivalTime);
    timeNextEvent[etDeparture]  = INF;
}

Simulation::~Simulation() {

}

void Simulation::run() {

	// print a pretty little header
    cout << "Single-server queueing system" << endl << endl;
    cout << "Mean interarrival time = " << meanInterarrivalTime << " minutes" << endl;
    cout << "Mean service time      = " << meanServiceTime << " minutes"<< endl;
    cout << "Number of customers    = " << totalCustomersDelayed << endl << endl;


	while (numberCustomersDelayed < totalCustomersDelayed) {
		timing();
		updateTimeStats();

		switch (nextEventType) {
			case etArrival: {
				arrival();
			} break;
			case etDeparture: {
				departure();
			} break;
			default: {
				throw("Unhandled Event Type!");
			}break;
		}
	}
}

void Simulation::arrival() {

    timeNextEvent[etArrival] = simulationTime + expon(meanInterarrivalTime);

    if (serverState == ssBusy) {
        waitQueue.push(simulationTime);
    } else {
        float delay = 0.0;
        totalTimeCustomersDelayed += delay;

        ++numberCustomersDelayed;
        serverState = ssBusy;

		timeNextEvent[etDeparture] = simulationTime + expon(meanServiceTime);
    }
}

void Simulation::departure() {

    if (waitQueue.size() == 0) {
        serverState                = ssIdle;
        timeNextEvent[etDeparture] = INF;
    } else {
        float delay = simulationTime - waitQueue.front();
		waitQueue.pop();
        totalTimeCustomersDelayed += delay;

        ++numberCustomersDelayed;
        timeNextEvent[etDeparture] = simulationTime + expon(meanServiceTime);
    }
}

void Simulation::timing() {
    int i;
    float minimumTimeNextEvent = INF;

    nextEventType = etDummy;

	for(i = 1; i <= numberEventType; ++i) {
        if(timeNextEvent[i] < minimumTimeNextEvent) {
            minimumTimeNextEvent = timeNextEvent[i];
            nextEventType        = (EventType)i;
        }
	}

    if(nextEventType == etDummy) {
		cout << "Event list empty at time " << simulationTime << endl;
        exit(1);
    }

	simulationTime = minimumTimeNextEvent;
}

void Simulation::reportStats() const {
	cout << "Average delay in queue  = " << totalTimeCustomersDelayed / numberCustomersDelayed << " minutes" << endl;
	cout << "Average number in queue = " << areaNumberInQueue / simulationTime << endl;
	cout << "Server utilization      = " << areaServerStatus / simulationTime << endl;
	cout << "Time simulation ended   = " << simulationTime << endl;
}

void Simulation::updateTimeStats() {
    float timeSinceLastEvent;

    timeSinceLastEvent = simulationTime - timeLastEvent;
    timeLastEvent      = simulationTime;

    areaNumberInQueue += waitQueue.size() * timeSinceLastEvent;

    areaServerStatus  += serverState * timeSinceLastEvent;
}

float Simulation::expon(const float mean) const {
    return(-mean * log(lcgrand(1)));
}
