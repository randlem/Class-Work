// RawRNG.h
//
// A class for random number generation that aggregates
// the NetRNG and LocalRNG behaviors.  The type of behavior
// is a construction-time parameter, so it's possible
// to have instances possessing both behaviors.
//
// Two static constants are defined that let the user select
// behavior, such as the following:
// 	RawRNG lRNG(RawRNG::Local);
// 	RawRNG nRNG(RawRNG::Net);
// Currently we default to the RawRNG::Local behavior.
//
// METHODS:
// --------
//
// RawRNG(unsigned short)
// Constructor; argument sets the instance behavior to Net
// or Local RNG.  Constructs the appropriate RNG for its
// behavior.
//
// ~RawRNG()
// Destructor; deallocates our captive RNG object.
//
// genRandReal1() : double
// Fetches a double on the interval [0,1] from the appropriate
// RNG object.
//
// seedRand(unsigned long)
// Seeds the random number generator, if the random number
// generator is seedable.

#ifndef RAWRNG
#define RAWRNG

class RawRNG
{
	public:
		virtual double genRandReal1() = 0;
		//virtual void seedRand(unsigned long);
	private:
};

#endif
