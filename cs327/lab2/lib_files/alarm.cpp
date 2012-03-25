/********************************************************
*  CS 327                                               *
*  alarm.cxx                                            *
*         Use of alarm                                  *
*   An alarm timer is set.  The program proceeds        *
*   when the timer expires, an alarm signal is sent to  *
*   the process.  The process catches the signal - it   *
*   is 'sent' to the processAlarm function for handling *
********************************************************/

#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream.h>
#include <stdlib.h>

int counter;
void processAlarm (int);

int main() {
        counter = 0;
        alarm(5);
        signal (SIGALRM, processAlarm);   
        while (1) {
                ++counter;
        }

}
void processAlarm (int value) /*  process alarm */
{
	cout << " enter alarm handler " << endl;
	cout << " arguement passed is " << value << endl;
        cout << " Counter is: " << counter << endl;
        exit(1);
}

