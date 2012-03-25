// File method0.cpp version of 15 October 2003
// All code and comments written by Walter Maner unless credited otheriwse.

#include <iostream>
#include <iomanip>
#include <math.h>
#include "Timer.h"

int charStringToInt( const char [] );

double doNothing( double arg ) {
    return arg;
}

int main( int numArgs, char *args[] ) {
    if ( numArgs != 2 ) {
	cout << "Usage:   method0 repetitions" << endl;
	cout << "Example: method0 1000000" << endl;
	return 1;
    }
    Timer aTimer1, aTimer2;
    const long int REPS = charStringToInt( args[ 1 ] );
    long double junk;
    
    // Control loop
    // This loop generates the overhead that we subtract out
    aTimer1.start();
    
    // Call to sqrt() is used to defeat optimizations.

    for ( long int i = 0; i < REPS; i++ ) {
	junk = sqrt( i );
    }
    aTimer1.stop();

    // Experimental loop
    // This loop makes the measurement of interest
    aTimer2.start();
    long int j;
    for ( long int i = 0; i < REPS; i++ ) {
	junk = sqrt( i );
	j = 45; 
    }
    aTimer2.stop();

    aTimer1.display( "aTimer1 info" );
    aTimer2.display( "atimer2.info" );
    
    cout
    << setiosflags( ios :: fixed | ios :: showpoint )
	// Change next line to agree with expression being tested	
        << "j = 45 takes" 
        << setw( 14 ) << setprecision( 11 )
        // Subtract overhead:
        << ( aTimer2 - aTimer1 ) / REPS 
        << " secs per rep" << endl << endl;
    
    cout << "REPS = " << REPS << endl;
    
    return 0;
}

int charStringToInt( const char charString[] ) {
    int i = 0, returnValue = 0;
    while ( charString[ i ] ) {
        returnValue = returnValue * 10 + charString[ i++ ] - 48;
    }
    return returnValue;
}


/* TYPICAL OUTPUT
 * 
 * ====== aTimer1 info ======
 * 1 of 2 Timer instances
 * resolution = 0.01 secs
 * elapsedTime = 391.96 secs
 * running = no
 * startTime.tms_utime = 1 ticks
 * stopTime.tms_utime = 39197 ticks
 * ==========================
 * 
 * 
 * ====== atimer2.info ======
 * 2 of 2 Timer instances
 * resolution = 0.01 secs
 * elapsedTime = 405.46 secs
 * running = no
 * startTime.tms_utime = 39197 ticks
 * stopTime.tms_utime = 79743 ticks
 * ==========================
 * 
 * j = 45 takes 0.00000001350 secs per rep
 * 
 * REPS = 1000000000
 */
