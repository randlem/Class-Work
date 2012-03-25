// NetRNG.h
//
// A class for random-number "generation" which fetches
// numbers from the random.org CGI application.
//
// This class is very complicated.  The best description
// I can give follows.
//
// NetRNG uses a static member variable to track active
// instances.  If a NetRNG being constructed is the first,
// a new thread of execution is created whose only job
// is to fill a bounded buffer with doubles on the
// interval [0,1] which are acquired by TCP connection with
// a random.org CGI app that generates true random
// numbers based on digitized radio-band noise.  When
// the last active NetRNG has its destructor invoked,
// the static stop() method is called to tear down the
// Bounded Buffer and collect the "fetcher" thread.
//
// Requires working PThread library!
//
// METHODS:
// --------
//
// NetRNG()
// Constructor; nothing to do unless we're the first object
// being created, in which case we start up a thread to
// fetch random numbers into a buffer for us.  Each object
// constructed increments the count of active RNGs.
//
// ~NetRNG()
// Destructor; nothing to do unless we're the last active
// object, in which case we tear down the buffer and collect
// the fetcher thread. Each object destroyed decrements the
// count of active RNGs.
//
// operator!() : bool
// Checks to see if NetRNG statics are in a state where we
// can get numbers from it.  Returns true if we are ready.
//
// genRandReal1() : double
// Return a double on the interval [0,1]; this call can
// block the calling thread if the buffer is empty.

#include <pthread.h>

#include "HTTPSocket.h"
#include "BoundedBuffer.h"
#include "RawRNG.h"

#ifndef NETRNG
#define NETRNG

class NetRNG : public RawRNG {
	public:
		NetRNG();
		~NetRNG();
		inline bool const operator!() { return !NetRNG::ready; }
		inline double genRandReal1(){
			double tempVal;
			numberBuffer.get(tempVal);
			return tempVal;
		}
	private:

		static bool fetchSet();
		static void * start(void *);
		static void stop();

		static pthread_t myThread;
		static pthread_mutex_t mutex;
		static bool ready;

		static unsigned short activeRNG;
		static HTTPSocket uplink;
		static BoundedBuffer<double> numberBuffer;
};

#endif
