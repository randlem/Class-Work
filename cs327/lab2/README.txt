README.txt
Mark Randles
CS327
Lab2

SIGNALS: 

 SIGTSTP:
    Interactive Stop.  I'm using this as the "default" signal to handle.  I chose
    this because it's trapable (unlike SIGSTOP and SIGKILL) and is enterable from 
    the terminal (without the utility kill).  The handler closes the read end of my
    pipe between the parent and child process which should render a SIGPIPE the next 
    time that the child trys to write to the pipe.
    
    Normal Operation: Sent by Ctrl-Z input from user or kill SIGTSTP.
    
    Forced Operation: I'm sending it by way of keyboard interaction.
    
 SIGHUP:
    Hangup (old skool). I'm using this as a signal for the parent process to terminate.
    This indirect method of termination was needed because the child process was being
    terminated (at least on BGUNIX) before it could write back to the pipe and recieve
    the SIGPIPE message (if parent is destroyed then child is destroyed).  I set it up
    so that the child must send SIGHUP to the parent after it's recieved a SIGPIPE.  It 
    probally would be just as efficient to send SIGKILL to the parent after SIGPIPE, but 
    this I believe is more in the spirit of the exercise (another signal for my program 
    to handle).
    
    Normal Operation: Usually sent to daemon processes to terminate or restart by way
        of the utility kill or a shell script.
    
    Forced Operation: Sent by the SIGPIPE handler to signal the parent process to exit.
    
 SIGPIPE:
    Broken Pipe. Sent to the process after a write to a pipe where the corresponding read
    end has been closed.  This message gets sent after the parent recieves a SIGSTSTP and 
    closes it's end of the read pipe, and the child tries a write to that pipe.  The way I
    handle it is to send SIGHUP to the parent and then terminate the child process with
    exit.

    Normal Operation: Sent when a write to a pipe fails.  Notifies process of a broken pipe.
    
    Forced Operation: Sent after an attemped write to the pipe after SIGTSTP closes it's read
        end.

NOTES:
    The screen output is there to show that the processea are running in the background.  The 
    word is of no significance, it's just fun to say!
        
EOF