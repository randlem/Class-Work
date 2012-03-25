/************************************************************************
 * FILE:       main.cpp
 * WRITTEN BY: Mark Randles
 * COURSE:     CS327
 * ASSIGNMENT: Lab Assignment 1
 * DUE DATE:   5:00PM Thursday, September 24
 *
 * OVERVIEW: 
 *
 * INPUT: Number of rand() calls to make.
 *
 * OUTPUT: The total user time, CPU time, and current time for both the parent
 *  and child processcies.
 *
 * FUNCTIONS:
 * 
 * main(int argv, char* argc[])
 *  Main entry point for the program.  This function performs all of the
 *  nessary operations for the program.
 *
 ***********************************************************************/

// INCLUDES *************************************************************
#include <iostream.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <memory.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

// NAMESPACES ***********************************************************

// MACROS ***************************************************************

// GLOBALS **************************************************************

// STRUCTS **************************************************************
typedef struct TIME_STATS {
    timeval cpu_time;
    timeval user_time;
    timeval curr_time;
}; // end struct TIME_STATS

// CLASSES **************************************************************

// PROTOTYPES ***********************************************************

// FUNCTIONS ************************************************************
// the one, the only, the main()!
int main(int argv, char argc[]) { 
    // variable identification and initlization
    int pid;                   // pid of the child process
    int iopipe[2];             // fifo pipe
    int reps = atoi(&argc[1]); // the number of reps to perform
    
    // create a named pipe
    pipe(iopipe);
    
    // fork the process
    pid = fork();
    
    // decide on what to do based on the pid
    if(pid == 0) {
        // variable identification and initlization
        rusage resource_usage;     // struct to get resourace usage    
        TIME_STATS times;          // struct to store the time stats for this process
        
        memset(&resource_usage,0,sizeof(rusage));
    
        // close the read pipe so i don't accidently do something stupid
        close(iopipe[0]);
           
        // seed the psudo-random number generator
        srand(reps);
            
        // perform the required number of rand() calls
        for(int i=0; i < reps; i++) {
            rand();
        }// end for i    
    
        // get the time usage to this point.
        if(getrusage(RUSAGE_SELF,&resource_usage) == -1) {
            exit(2);
        }

        // fill the time stats struct
        memcpy(&times.cpu_time,&resource_usage.ru_stime,sizeof(timeval));
        memcpy(&times.user_time,&resource_usage.ru_utime,sizeof(timeval));
        gettimeofday(&times.curr_time,NULL);
        
        // write out the struct to the pipe
        write(iopipe[1],&times,sizeof(TIME_STATS));
        close(iopipe[1]);
        
        exit(1);
        
    } else if(pid != -1) {
        // variable identification and initlization
        rusage resource_usage;     // struct to get resource usage
        timeval curr_time;         // struct to get the current time of day
        TIME_STATS child_times;    // struct to hold the time stats about the child process
        
        // close the write pipe so i don't accidently do something stupid
        close(iopipe[1]);
    
        // read in the resource usage struct
        read(iopipe[0],&child_times,sizeof(TIME_STATS));
        close(iopipe[0]);
    
        // output the child information
        cout << "Child User Time: " << child_times.user_time.tv_sec << " " << (child_times.user_time.tv_usec) << endl;
        cout << "Child CPU Time: " << child_times.cpu_time.tv_sec << " " << (child_times.cpu_time.tv_usec) << endl;        
        cout << "Child Current Time: " << child_times.curr_time.tv_sec << " " << (child_times.curr_time.tv_usec) << endl;
        
        // get the parent information
        if(getrusage(RUSAGE_CHILDREN,&resource_usage) == -1) {
            return(1);
        }
        
        // get the parent current time of day
        gettimeofday(&curr_time,NULL);
        
        // output the parent information
        cout << "Parent User Time: " << resource_usage.ru_utime.tv_sec << " " << (resource_usage.ru_utime.tv_usec) << endl;
        cout << "Parent CPU Time: " << resource_usage.ru_stime.tv_sec << " " << (resource_usage.ru_stime.tv_usec)<< endl;        
        cout << "Parent Current Time: " << curr_time.tv_sec << " " << curr_time.tv_usec << endl;
        
    } else {
        cout << "ERROR" << endl;
    }
            
    // return back to the system
    return(0);
        
}// end main()
