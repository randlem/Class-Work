/************************************************************************
 * FILE:       main.cpp
 * WRITTEN BY: Mark Randles
 * COURSE:     CS335
 * ASSIGNMENT: Lab Assignment 1
 * DUE DATE:   11:59PM Saturday, Sept 13
 *
 * OVERVIEW: A simple simulation.  The input queue is priority based.  Each
 *  object has two times associated with it X is the time to process the object
 *  in the "processing station" and Y is the time between object loads into the 
 *  input queue.  After processing the object get put in a output queue where they
 *  expire.  This is not a real-time simulation.  Each "clock" we will test the Y 
 *  elapsed and if it passes the threashold we will read in the next object.  Then
 *  the X elapsed time will be tested and if it passes the threshold the current
 *  object will be swaped out of the "processing station" and the highest priority
 *  object in the input queue will be swapped in.  Every time a object is swapped
 *  into the processing station a line of output will be written.
 *
 * INPUT: sim.in
 *
 * OUTPUT: List of object name, priorities, processing time, and read time for
 *         each object, to stdout.
 *
 * FUNCTIONS:
 * 
 * main(int argv, char* argc[])
 *  Main entry point for the program.  Performs the main timing loop.
 * 
 ***********************************************************************/

// INCLUDES *************************************************************
#include <queue>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <iostream.h>
#include <fstream.h>

// NAMESPACES ***********************************************************
using namespace std;

// MACROS ***************************************************************

// STRUCTS **************************************************************
typedef struct TIMER {
    int threshold;
    int value;
    bool operator ++ (int) {
        value++;
        return(value >= threshold);
    }
};

// CLASSES **************************************************************
class Task {
    public:
        Task() { m_empty = true; }
        Task(char name[80], int priority, int proc_time) { 
            memset(m_name,0,sizeof(char)*80);
            strcpy(m_name,name);
            m_priority = priority; 
            m_proc_timer.threshold = proc_time;
            m_proc_timer.value = 0;
            m_empty = false;
        }
        Task(char* input) {
            memset(m_name,0,sizeof(char)*80);
            strcpy(m_name,strtok(input," "));
            m_priority = atoi(strtok(NULL," "));
            m_proc_timer.threshold = atoi(strtok(NULL," "));
            m_empty = false;
        }
        Task(fstream& input) {
            memset(m_name,0,sizeof(char)*80);
            input >> m_name;
            input >> m_priority;
            input >> m_proc_timer.threshold;
            m_proc_timer.value = 0;
            m_empty = false;
        }
        ~Task() { }
        
        void Status(char* buffer) {
            sprintf(buffer,"%s: priority=%i proc_time=%i %i",m_name,m_priority,m_proc_timer.threshold,m_proc_timer.value);
        }
        
        bool operator < ( const Task & aTask ) const {
            return (m_priority < aTask.m_priority);
        }
        
        void operator = (const Task& aTask) {
            memset(m_name,0,sizeof(char)*80);
            strcpy(m_name,aTask.m_name);
            m_priority = aTask.m_priority;
            m_proc_timer.value = 0;
            m_proc_timer.threshold = aTask.m_proc_timer.threshold;
            m_empty = aTask.m_empty;
        }
        
        bool Process() {
           return(m_proc_timer++);
        }
        
        bool Empty() {
            return(m_empty);
        }
        
        void Delete() {
            memset(m_name,0,sizeof(char)*80);
            m_empty = true;
        }
        
    protected:
    
    private:
        char m_name[80];
        int m_priority;
        TIMER m_proc_timer;
        bool m_empty;
};

// PROTOTYPES ***********************************************************

// FUNCTIONS ************************************************************
// the one, the only, the main()!
int main(int argv, char argc[]) {
    
    // variable identification and initlization
    fstream input;                 // file stream for the input
    int ticks;                     // the current clock tick
    TIMER input_timer;             // timer for the input stream
    char buffer[80];               // input character buffer
    char output[255];              // output character buffer
    Task curr_task;                // pointer to the current task
    priority_queue<Task> input_queue; // the priority input queue
    queue<Task> output_queue;      // output queue
       
    // open the input file
    input.open("./sim.in",fstream::in);
    
    if(!input.is_open()) {
        cout << "Error opening input file.  Perhaps you don't have a sim.in?" << endl;
        exit(1);
    }
    
    // main cycle loop
    input_timer.value = 0; input_timer.threshold = 0;
    ticks = 0;
    while(true) {
        
        // test to see if the input cycle is up
        if((input_timer++) == true && !input.eof()) {
            input_timer.threshold = 0; 

            while(input_timer.threshold == 0 && !input.eof()) {
                memset(buffer,0,sizeof(char)*80);
        
                if(!input.eof()) { 
                    input_queue.push(Task(input));
                 }
            
                memset(buffer,0,sizeof(char)*80);
                input >> buffer;
                if(!input.eof()) {
                    input_timer.threshold = atoi(buffer);
                    input_timer.value = 0;
                }
            }

        }

        // test to see if the processing cycle is up               
        if(curr_task.Empty()) {
            if(!input_queue.empty()) {      
                output_queue.push(curr_task);
                curr_task = input_queue.top();
                input_queue.pop();
                memset(output,0,sizeof(char)*255);
                curr_task.Status(output);
                cout << ticks << ": input_queue=" << input_queue.size() << " output_queue=" << output_queue.size() << " " << output << endl;
            }
        } else {
            if(curr_task.Process()) {
                curr_task.Delete();
            }
        }

        if(input.eof() && input_queue.empty() && curr_task.Empty()) {
            break;
        }
        
        ticks++;

    }// end while    

    input.close();
    
    // exit back to the os
    exit(0);

}// end main()
 
 
