/***********************************************************
 * Lab 2 - CS327
 * Mark Randles
 * Instructor: G. Zimmerman
 * Class: CS327 8:00AM-9:15AM T R
 *
 * Objective: Write a program that consists of two threads of 
 *  execution.  Each thread is to handle the SIGPIPE and one 
 *  other signal of my choice.  I will be catching the Ctrl+Z 
 *  (SIGSTOP) signal.  This should be entered from the terminal
 *  or sent when the child exits.  The child will write a "status"
 *  to the pipe to genereate the SIGPIPE signal.  The parent will
 *  process the child "status" and perhaps send it the SIGSTOP if
 *  it the child should exit.  The same applys for the SIGSTOP 
 *  signal from the CLI.  If SIGSTOP is recieved in the parent
 *  then the parent will send SIGSTOP to the child and wait for 
 *  the child to resend SIGSTOP before exiting.  This ensures 
 *  that the child exited before the parent.  Each signal 
 *  handler will output the meaning of the signal in it's context,
 *  and the perform the nessary processing of that signal for the 
 *  context of my program.
 *
 * Files:
 *  lab2.cpp - main source file
 *  README.txt - describes the signal processing of the program.
 *
 ***********************************************************/


#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h> 
#include <memory.h>

int p[2];            // pipe between parent and child
int pid;             // pid of process (0 if child)

void ParentSIGTSTP(int signal);
void ParentSIGHUP(int signal);
void ChildSIGPIPE(int signal);

int main(int argv, char* argc[]) {
    signal(SIGPIPE,ChildSIGPIPE);
    signal(SIGTSTP,ParentSIGTSTP);
    
    // make the pipe    
    pipe(p);
    
    // fork the process
    pid = fork();

    if(pid == 0) { // child process
        // close the read pipe as to not do anything dumb
        close(p[0]);

        // loop indefinatly writing gibbrish to the pipe
        char buffer[80] = "w00t!";
        write(p[1],buffer,strlen(buffer));
        
        while(1) { write(p[1],buffer,strlen(buffer)); }
        
    } else { // parent process
        // set the parent signal handlers
        signal(SIGHUP,ParentSIGHUP);
        
        // close the write pipe as to not do anything dumb
        close(p[1]);
                        
        // get the first write from the child process...blocking till i know that the child is running
        char buffer[81];
        memset(buffer,0,sizeof(char)*81); read(p[0],buffer,sizeof(char)*80); cout << buffer << endl;
        
        while(1) { sleep(1); memset(buffer,0,sizeof(char)*81); read(p[0],buffer,sizeof(char)*80); cout << buffer << endl;}
        
    }
    
    // exit back to the system with an error
    exit(1);
    
}

void ParentSIGTSTP(int signal) {
    // handles the SIGSTP for the parent.
    
    // status message
    cout << "Parent/Child: SIGTSTP" << endl;
    cout << "\tA stop signal was sent to the parent from the console.  We will clean up all the process here.  To terminate the child" << endl <<
            "\t we are going to close the pipe, therefore sending SIGPIPE the next time the child writes to the pipe.  This will cause" << endl <<
            "\t the child to send us back a SIGHUP to signal the parent to exit.  This was the best hack i could figure to get around " << endl <<
            "\t the parent exiting before the child and closing the stdout file stream before the child could write it\'s SIGPIPE stuff" << endl << 
            "\t to stdout.  It is repeated twice because it\'s caught by both the parent and child processes." << endl;
    cout << "Process pid: " << getpid() << endl << endl;
    close(p[0]);
}

void ParentSIGHUP(int signal) {
    // handles the SIGSTP for the parent.
    
    // status message
    cout << "Parent: SIGHUP" << endl;
    cout << "\tA SIGHUP was sent to the parent from the child.  I\'m using this message to tell the parent to now exit.  I\'ve done it" << endl <<
            "\tthis way to keep make sure the child\'s signal output get printed to stdout before the parent process is destroyed." << endl;
    cout << "Process pid: " << getpid() << endl << endl;
    exit(0);
}

void ChildSIGPIPE(int signal) {
    // Handles the SIGPIPE for the parent.
    
    // status message
    cout << "Child: SIGPIPE" << endl;
    cout << "\tA broken write pipe was found.  The parent process must have closed it\'s end of the pipe after recieving a SIGTSTP." << endl << 
            "\tI\'ll close my end of the broken pipe and gracefully exit." << endl;
    cout << "Process pid: " << getpid() << endl << endl;
    kill(getppid(),SIGHUP);
            
    // exit back to the system
    exit(0);
}
