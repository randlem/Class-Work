/********************************************************
*       C++ programs end with filname.cpp               * 
*       To compile                                      *
*       g++ filename.cpp
*       a.out                                           *
********************************************************/

//   This program illustrates the use of the system calls:
//      fork, getpid and exit
//   The original process forks creating a child process
//   The fork command returns the child process id to the
//   parent's pid variable; the childs pid variable is 0 

#include <iostream.h>
#include <unistd.h>

int number;
int main() {

 int pid;

 number = 1;
 pid =fork();

 if (pid == 0) {
        /* child */ 
	cout << " I am the Child.  My id is " << getpid() << 
                " Number is " << number << endl;
	cout << " Child computation " << endl;

        exit(1);
  } 
   
 else {      /* parent */
        number = 100;
	cout << " I am the Parent.  My id is " << getpid() << 
                " Number is " << number << endl;
	cout << " I am the Parent.  My child has id " << pid << 
                " Number is " << number << endl;
      } 
}
