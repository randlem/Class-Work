#include <vector>
using std::vector;

#include <stack>
using std::stack;

#include <iostream>
using std::cout;
using std::endl;

#ifndef RANDGEN_H
#define RANDGEN_H

class RandGen {
public:
	RandGen(int size) : expand(size) {
		this->seed = 0;
		this->used = 0;
		populateList(size);
		position = randList.begin();
	}

	RandGen(int size, int seed) : expand(size) {
		this->seed = seed;
		this->used = 0;
		populateList(size);
		position = randList.begin();
	}

	bool rewind(double t) {
		if(times.empty())
			return(true);

		while(!times.empty() && times.top() > t) {
			position--;
			used--;
			times.pop();
		}

		if(!times.empty()) {
			position--;
			used--;
			times.pop();
		}

		if(position <= randList.begin())
			position = randList.begin();
		return(true);
	}

	bool rewind(int n) {
		if(times.empty())
			return(true);

		for(int i=0; i < n-1; ++i) {
			position--;
			used--;
			times.pop();
			if(times.empty())
				return(true);
		}

		position--;
		used--;
		times.pop();

		if(position <= randList.begin())
			position = randList.begin();
		return(true);
	}

	float getRandom(double t) {
		++position;
		++used;
		times.push(t);
		if(position != randList.end())
			return(*(position));
		else {
			int offset = randList.size();
			populateList(expand);
			position = randList.begin() + offset;
		}
		return(*(position));
	}

	float peekRandom() {
		return(*(position+1));
	}

	int getIndex() {
		return(used);
	}

private:
	vector<float> randList;
	vector<float>::iterator position;
	stack<double> times;

	int seed;
	int expand;
	int used;

	RandGen() { ; }

	void populateList(int count) {
		for(int i=0; i < count; ++i)
			randList.push_back((float)genRand());
	}

	double genRand() {
		/*-------------------------------------------------------------------*/
		/* A C-program for TT800 : July 8th 1996 Version */
		/* by M. Matsumoto, email: matumoto@math.keio.ac.jp */
		/* genrand() generate one pseudorandom number with double precision */
		/* which is uniformly distributed on [0,1]-interval */
		/* for each call.  One may choose any initial 25 seeds */
		/* except all zeros. */

		/* See: ACM Transactions on Modelling and Computer Simulation, */
		/* Vol. 4, No. 3, 1994, pages 254-266. */

		const int NRan = 25;
		int MRan = seed % NRan;

		unsigned long y;
 		static int k = 0;
		static unsigned long x[NRan]={ /* initial 25 seeds, change as you wish */
                                     0x95f24dab, 0x0b685215, 0xe76ccae7, 0xaf3ec239, 0x715fad23,
                                     0x24a590ad, 0x69e4b5ef, 0xbf456141, 0x96bc1b7b, 0xa7bdf825,
                                     0xc1de75b7, 0x8858a9c9, 0x2da87693, 0xb657f9dd, 0xffdc8a9f,
                                     0x8121da71, 0x8b823ecb, 0x885d05f5, 0x4e20cd47, 0x5a9ad5d9,
                                     0x512c0c03, 0xea857ccd, 0x4cc1d30f, 0x8891a8a1, 0xa6b7aadb
                                 };
		static unsigned long mag01[2]={0x0, 0x8ebfd028 /* this is magic vector `a', don't change */};
		if (k==NRan) { /* generate NRan words at one time */
			int kk;
			for (kk=0;kk<NRan-MRan;kk++) {
				x[kk] = x[kk+MRan] ^ (x[kk] >> 1) ^ mag01[x[kk] % 2];
			}
			for (; kk<NRan;kk++) {
				x[kk] = x[kk+(MRan-NRan)] ^ (x[kk] >> 1) ^ mag01[x[kk] % 2];
			}
			k=0;
		}
		y = x[k];
		y ^= (y << 7) & 0x2b5b2500; /* s and b, magic vectors */
		y ^= (y << 15) & 0xdb8b0000; /* t and c, magic vectors */
		y &= 0xffffffff; /* you may delete this line if word size = 32 */
		/*
		   the following line was added by Makoto Matsumoto in the 1996 version
		   to improve lower bit's corellation.
		   Delete this line to o use the code published in 1994.
		*/
		y ^= (y >> 16); /* added to the 1994 version */
		k++;
		return( (double) y / (unsigned long) 0xffffffff);
	}

};

#endif

