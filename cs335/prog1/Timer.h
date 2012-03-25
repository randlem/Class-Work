// Timer.h version of 24 March 2003
// All code and comments written by Walter Maner unless otherwise noted

#ifndef TIMER_H
#define TIMER_H

#include <sys/times.h> // for tms
#include <unistd.h>    // for synconf()
#include <iostream>

class Timer {
        // This class is implementation and system dependent.
    public:
        Timer();
        Timer( Timer & aTimer );
        ~Timer();
        void start();                          // OK if not running
        void stop();                           // OK if running
        void clear();                          // OK if not running
        void display( char * caption ) const;  // OK if not running
        long double getResolution() const;     // OK anytime
        long double getElapsedTime() const;    // OK if not running
        long double operator-                  // OK if not running
        ( const Timer & aTimer ) const;
        long double operator+                  // OK if not running
        ( const Timer & aTimer ) const;
        bool operator<                         // OK if not running
        ( const Timer & aTimer ) const;
        bool operator>                         // OK if not running
        ( const Timer & aTimer ) const;
        bool operator==                        // OK if not running
        ( const Timer & aTimer ) const;
        bool operator!=                        // OK if not running
        ( const Timer & aTimer ) const;

    private:
        struct tms startTime, stopTime;
        long double elapsedTime;    // in seconds
        bool running;
        long double resolution; // in seconds
        static int instanceCount;
        int instanceNumber;

        void warning( char * message ) const;
        void setResolution();
};

Timer :: Timer() {
    instanceNumber = ++instanceCount;
    running = false;
    setResolution();
    clear();

}

Timer :: Timer( Timer & aTimer ) {
    if ( running || aTimer.running )
        warning( "Timer(s) running in Timer::Timer( Timer & )" );
    startTime = aTimer.startTime;
    stopTime = aTimer.stopTime;
    elapsedTime = aTimer.elapsedTime;
    running = aTimer.running;
    resolution = aTimer.resolution;
    instanceNumber = aTimer.instanceNumber;
}


Timer :: ~Timer() {}

void Timer :: start() {
    if ( running )
        warning( "Timer already running in Timer::start()" );
    running = true;
    if ( times( &startTime ) == -1 )
        warning( "Bad return value from times() in Timer::start()" );
}

void Timer :: stop() {
    if ( times( &stopTime ) == -1 )
        warning( "Bad return value from times() in Timer::stop()" );
    if ( !( running ) )
        warning( "Timer not running before Timer::stop()" );
    running = false;
    elapsedTime = resolution * ( stopTime.tms_utime - startTime.tms_utime );
}

void Timer :: clear() {
    if ( running )
        warning( "Timer still running in Timer::clear()" );
    elapsedTime = 0.0;
}

void Timer :: display( char * caption ) const {
    int lineLength = strlen( caption ) + 2 * strlen( " ======" );
    if ( running )
        warning( "Timer running in Timer::display()" );
    cout
    << endl
    << "====== " << caption << " ======" << endl
    << instanceNumber << " of " << instanceCount << " Timer instances" << endl
    << "resolution = " << resolution << " secs" << endl
    << "elapsedTime = " << elapsedTime << " secs" << endl
    << "running = ";
    if ( running )
        cout << "yes" << endl;
    else
        cout << "no" << endl;
    cout
    << "startTime.tms_utime = " << startTime.tms_utime << " ticks" << endl
    << "stopTime.tms_utime = " << stopTime.tms_utime << " ticks" << endl;
    for ( int i = 0; i < lineLength; i++ )
        cout << "=";
    cout << endl << endl;

}


long double Timer :: getResolution () const {
    return resolution;
}

long double Timer :: operator- ( const Timer & aTimer ) const {
    if ( running || aTimer.running )
        warning( "Timer(s) still running in Timer::operator-" );
    return this->elapsedTime - aTimer.elapsedTime;
}

long double Timer :: operator+ ( const Timer & aTimer ) const {
    if ( running || aTimer.running )
        warning( "Timer(s) still running in Timer::operator+" );
    return this->elapsedTime + aTimer.elapsedTime;
}

bool Timer :: operator< ( const Timer & aTimer ) const {
    if ( running || aTimer.running )
        warning( "Timer(s) still running in Timer::operator<" );
    return this->elapsedTime < aTimer.elapsedTime;
}

bool Timer :: operator> ( const Timer & aTimer ) const {
    if ( running || aTimer.running )
        warning( "Timer(s) still running in Timer::operator>" );
    return this->elapsedTime > aTimer.elapsedTime;
}

bool Timer :: operator== ( const Timer & aTimer ) const {
    if ( running || aTimer.running )
        warning( "Timer(s) still running in Timer::operator==" );
    return this->elapsedTime == aTimer.elapsedTime;
}

bool Timer :: operator!= ( const Timer & aTimer ) const {
    if ( running || aTimer.running )
        warning( "Timer(s) still running in Timer::operator!=" );
    return this->elapsedTime != aTimer.elapsedTime;
}

long double Timer :: getElapsedTime() const {
    if ( running )
        warning( "Timer still running in Timer::getElapsedTime()" );
    return elapsedTime;
}

void Timer :: warning( char * message ) const {
    cout << endl << "Warning: " << message << endl;
}

void Timer :: setResolution() {
    resolution = 1.0 / sysconf( _SC_CLK_TCK );  // Implementation dependent
}

int Timer :: instanceCount = 0;

#endif
