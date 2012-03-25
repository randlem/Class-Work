/********************************************************
*         system calls: use Pipe                        *       
*         Parent reads, Child writes                    *
********************************************************/
#include <iostream.h>
#include <unistd.h>

int main() {
 char buffer[512], bufferParent[ 512];
 int pid, pipeName[2];

 pipe (pipeName);  /* [0] is 'read' end; [1] is 'write' end */
 pid = fork();

 if (pid == 0) {

        /* child */ 
        cout << "I am the child; my ID" << getpid() << " "<<  endl;
        cout << "Please enter your first name" << endl;

        cin >> buffer;                          // read from the terminal window
        cout << "Child  printing..." << buffer << endl; // print to term window

        close (pipeName[0]);    //I don't read off of pipe
        write (pipeName[1], buffer, strlen(buffer)+1);
        close (pipeName[1]);    //all done with writing
        exit(1);

  } 
   
 else{      /* parent */
        cout << "I am the parent; my ID = " << getpid() << endl;
        close (pipeName[1]);    //I don't  write to pipe

        read (pipeName[0], bufferParent, 512);
        close (pipeName[0]);    //all done with reading 
        cout << "parent printing..." << bufferParent << endl;  
                // print to term window



 } 
}



