#include <iostream>
using std::cout;
using std::endl;

#include <queue>
using std::queue;

#include "lcgrand.h"

#ifndef SIMULATION_H
#define SIMULATION_H

enum EventType {etDummy=0, etArrival=1, etDeparture};

const float INF = 1.0e+30;
const int numberEventType = 2;

class Simulation {
public:
	Simulation(int, float, float);
	~Simulation();

	void run();
	void reportStats() const;

private:
	enum ServerState {ssIdle, ssBusy};

	Simulation();

	void arrival();
	void departure();
	void timing();
	float expon(const float) const;
	void updateTimeStats();

	// master event time
	float simulationTime;

	// values that the user can set
	float meanInterarrivalTime;
	float meanServiceTime;
	int totalCustomersDelayed;

	// parts of the server
	ServerState serverState;
	queue<float> waitQueue;

	// event time variables
	float timeLastEvent;
	float timeNextEvent[3];
	EventType nextEventType;

	// stats stuff
	int numberCustomersDelayed;
    float totalTimeCustomersDelayed;
    float areaNumberInQueue;
    float areaServerStatus;
};

#endif
