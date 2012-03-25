#include <stdlib.h>
#include <stdio.h>

#include "simlib.h"

/* event defines */
#define EVENT_ADMIN_ARRIVAL   1
#define EVENT_ADMIN_SERVICE   2
#define EVENT_NURSE0_ARRIVAL  3
#define EVENT_NURSE1_ARRIVAL  4
#define EVENT_NURSE0_SERVICE  5
#define EVENT_NURSE1_SERVICE  6
#define EVENT_DOCTOR0_ARRIVAL 7
#define EVENT_DOCTOR1_ARRIVAL 8
#define EVENT_DOCTOR0_SERVICE 9
#define EVENT_DOCTOR1_SERVICE 10

/* defines for the list numbers for various queues and servers */
#define LIST_ADMIN_QUEUE    1
#define LIST_ADMIN_SERVER   2
#define LIST_NURSE0_QUEUE   3
#define LIST_NURSE1_QUEUE   4
#define LIST_NURSE0_SERVER  5
#define LIST_NURSE1_SERVER  6
#define LIST_DOCTOR0_QUEUE  7
#define LIST_DOCTOR1_QUEUE  8
#define LIST_DOCTOR0_SERVER 9
#define LIST_DOCTOR1_SERVER 10

/* random stream times for arrival/service for all the
   different services */
#define STREAM_ADMIN_ARRIVAL   1
#define STREAM_ADMIN_SERVICE   2
#define STREAM_NURSE0_ARRIVAL  3
#define STREAM_NURSE1_ARRIVAL  4
#define STREAM_NURSE0_SERVICE  5
#define STREAM_NURSE1_SERVICE  6
#define STREAM_DOCTOR0_ARRIVAL 7
#define STREAM_DOCTOR1_ARRIVAL 8
#define STREAM_DOCTOR0_SERVICE 9
#define STREAM_DOCTOR1_SERVICE 10
#define STREAM_NURSE_PROB      11
#define STREAM_DOCTOR_PROB     12

/* sampst defines */
#define SAMPST_ADMIN_DELAY   1
#define SAMPST_NURSE0_DELAY  2
#define SAMPST_NURSE1_DELAY  3
#define SAMPST_DOCTOR0_DELAY 4
#define SAMPST_DOCTOR1_DELAY 5

void init_model(void);

void admin_arrival(void);
void admin_departure(voidD);
void nurse0_arrival(void);
void nurse0_departure(void);
void nurse1_arrival(void);
void nurse1_departure(void);
void doctor0_arrival(void);
void doctor0_departure(void);
void doctor1_arrival(void);
void doctor1_departure(void);

int num_patients_depart,num_patients_required=1000;
float mean_admin_arrival=1,mean_admin_service=1;
float mean_nurse_arrival=.5,mean_nurse_service=5;
float mean_doctor_arrival=5,mean_doctor_service=5;

int main(int argc, char* argv[]) {

	init_simlib();
	maxatr = 4;

	init_model();

	while(num_patients_depart < num_patients_required) {
		timing();

		fflush(stdout);
		switch(next_event_type) {
		case EVENT_ADMIN_ARRIVAL: {
			printf("admin_arrival()\n");
			admin_arrival();
		}break;
		case EVENT_ADMIN_SERVICE: {
			printf("admin_departure()\n");
			admin_departure();
		}break;
		case EVENT_NURSE0_ARRIVAL: {
			printf("nurse0_arrival()\n");
			nurse0_arrival();
		}break;
		case EVENT_NURSE0_SERVICE: {
			printf("nurse0_departure()\n");
			nurse0_departure();
		}break;
		case EVENT_NURSE1_ARRIVAL: {
			printf("nurse1_arrival()\n");
			nurse1_arrival();
		}break;
		case EVENT_NURSE1_SERVICE: {
			printf("nurse1_departure()\n");
			nurse1_departure();
		}break;
		case EVENT_DOCTOR0_ARRIVAL: {
			printf("doctor0_arrival()\n");
			doctor0_arrival();
		}break;
		case EVENT_DOCTOR0_SERVICE: {
			printf("doctor0_departure()\n");
			doctor0_departure();
		}break;
		case EVENT_DOCTOR1_ARRIVAL: {
			printf("doctor1_arrival()\n");
			doctor1_arrival();
		}break;
		case EVENT_DOCTOR1_SERVICE: {
			printf("doctor1_departure()\n");
			doctor1_departure();
		}break;
		default:fprintf(stderr,"INVALID EVENT TYPE");exit(1);break;
		}
	}

	return(0);
}

void init_model(void)  /* Initialization function. */
{
	num_patients_depart = 0;

	event_schedule(sim_time + expon(mean_admin_arrival, STREAM_ADMIN_ARRIVAL),
                   EVENT_ADMIN_ARRIVAL);
}

void admin_arrival(void) {
	/* schedule the next arrival event
	event_schedule(sim_time + expon(mean_admin_arrival, STREAM_ADMIN_ARRIVAL),
	                 EVENT_ADMIN_ARRIVAL);
	                 
	/* see if the list is empty */
	if(list_size[LIST_ADMIN_SERVER] == 1) {
		transfer[1] = sim_time;
		list_file(LAST,LIST_ADMIN_QUEUE);
	} else {
		sampst(0.0,SAMPST_ADMIN_DELAY);
		list_file(FIRST,LIST_ADMIN_SERVER);
		event_schedule(sim_time+expon(mean_admin_service, STREAM_ADMIN_SERVICE),
										 EVENT_ADMIN_SERVICE);
	}
}

void admin_departure(voidD) {
	if(lcgrand(STREAM_NURSE_PROB) >= .5) {
		event_schedule(sim_time+expon(mean_nurse_arrival,STREAM_NURSE1_ARRIVAL),
										EVENT_NURSE1_ARRIVAL);
	} else {
		event_schedule(sim_time+expon(mean_nurse_arrival,STREAM_NURSE0_ARRIVAL),
										EVENT_NURSE0_ARRIVAL);
	}

	if(list_size[LIST_ADMIN_QUEUE] == 0) {
		list_remove(FIRST,LIST_ADMIN_SERVER);
	} else {
		list_remove(FIRST,LIST_ADMIN_QUEUE);
		sampst(sim_time - transfer[1],SAMPST_ADMIN_DELAY);
		event_schedule(sim_time+expon(mean_admin_service,STREAM_ADMIN_SERVICE),
										EVENT_ADMIN_SERVICE);
	}
}

void nurse0_arrival(void) {
	if(list_size[LIST_NURSE0_SERVER] == 1) {
		transfer[1] = sim_time;
		list_file(LAST,LIST_NURSE0_QUEUE);
	} else {
		sampst(0.0,SAMPST_NURSE0_DELAY);
		list_file(FIRST,LIST_NURSE0_SERVER);
		event_schedule(sim_time+expon(mean_nurse_service, STREAM_NURSE0_SERVICE),
										 EVENT_NURSE0_SERVICE);
	}
}

void nurse0_departure(void) {
	if(lcgrand(STREAM_DOCTOR_PROB) >= .1) {
		event_schedule(sim_time+expon(mean_doctor_arrival,STREAM_DOCTOR0_ARRIVAL),
										EVENT_DOCTOR0_ARRIVAL);
	} else {
		event_schedule(sim_time+expon(mean_doctor_arrival,STREAM_DOCTOR1_ARRIVAL),
										EVENT_DOCTOR1_ARRIVAL);
	}

	if(list_size[LIST_NURSE0_QUEUE] == 0) {
		list_remove(FIRST,LIST_NURSE0_SERVER);
	} else {
		list_remove(FIRST,LIST_NURSE0_QUEUE);
		sampst(sim_time - transfer[1],SAMPST_NURSE0_DELAY);
		event_schedule(sim_time+expon(mean_nurse_service,STREAM_NURSE0_SERVICE),
										EVENT_NURSE0_SERVICE);
	}
}

void nurse1_arrival(void) {
	if(list_size[LIST_NURSE1_SERVER] == 1) {
		transfer[1] = sim_time;
		list_file(LAST,LIST_NURSE1_QUEUE);
	} else {
		sampst(0.0,SAMPST_NURSE1_DELAY);
		list_file(FIRST,LIST_NURSE1_SERVER);
		event_schedule(sim_time+expon(mean_nurse_service, STREAM_NURSE1_SERVICE),
										 EVENT_NURSE1_SERVICE);
	}
}

void nurse1_departure(void) {
	if(lcgrand(STREAM_DOCTOR_PROB) >= .1) {
		event_schedule(sim_time+expon(mean_doctor_arrival,STREAM_DOCTOR0_ARRIVAL),
										EVENT_DOCTOR0_ARRIVAL);
	} else {
		event_schedule(sim_time+expon(mean_doctor_arrival,STREAM_DOCTOR1_ARRIVAL),
										EVENT_DOCTOR1_ARRIVAL);
	}
	
	if(list_size[LIST_NURSE1_QUEUE] == 0) {
		list_remove(FIRST,LIST_NURSE1_SERVER);
	} else {
		list_remove(FIRST,LIST_NURSE1_QUEUE);
		sampst(sim_time - transfer[1],SAMPST_NURSE1_DELAY);
		event_schedule(sim_time+expon(mean_nurse_service,STREAM_NURSE1_SERVICE),
										EVENT_NURSE1_SERVICE);
	}
}

void doctor0_arrival(void) {
	if(list_size[LIST_DOCTOR0_SERVER] == 1) {
		transfer[1] = sim_time;
		list_file(LAST,LIST_DOCTOR0_QUEUE);
	} else {
		sampst(0.0,SAMPST_DOCTOR0_DELAY);
		list_file(FIRST,LIST_DOCTOR0_SERVER);
		event_schedule(sim_time+expon(mean_nurse_service, STREAM_DOCTOR0_SERVICE),
										 EVENT_DOCTOR0_SERVICE);
	}
}

void doctor0_departure(void) {
	if(list_size[LIST_DOCTOR0_QUEUE] == 0) {
		list_remove(FIRST,LIST_DOCTOR0_SERVER);
	} else {
		list_remove(FIRST,LIST_DOCTOR0_QUEUE);
		++num_patients_depart;		
		sampst(sim_time - transfer[1],SAMPST_DOCTOR0_DELAY);
		event_schedule(sim_time+expon(mean_doctor_service,STREAM_DOCTOR0_SERVICE),
										EVENT_DOCTOR0_SERVICE);
	}
}

void doctor1_arrival(void) {
	if(list_size[LIST_DOCTOR1_SERVER] == 1) {
		transfer[1] = sim_time;
		list_file(LAST,LIST_DOCTOR1_QUEUE);
	} else {
		sampst(0.0,SAMPST_DOCTOR1_DELAY);
		list_file(FIRST,LIST_DOCTOR1_SERVER);
		event_schedule(sim_time+expon(mean_doctor_service, STREAM_DOCTOR1_SERVICE),
										 EVENT_DOCTOR1_SERVICE);
	}
}

void doctor1_departure(void) {
	if(list_size[LIST_DOCTOR1_QUEUE] == 0) {
		list_remove(FIRST,LIST_DOCTOR1_SERVER);
	} else {
		list_remove(FIRST,LIST_DOCTOR1_QUEUE);
		++num_patients_depart;
		sampst(sim_time - transfer[1],SAMPST_DOCTOR1_DELAY);
		event_schedule(sim_time+expon(mean_nurse_service,STREAM_DOCTOR1_SERVICE),
										EVENT_DOCTOR1_SERVICE);
	}
}

