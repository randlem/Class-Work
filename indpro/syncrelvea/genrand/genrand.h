#include <stdio.h>
#include <memory.h>

#ifndef __GENRAND_H__
#define __GENRAND_H__

#define MIN_LIST_SIZE 10
#define LIST_SIZE_INC 10

void initRandGenEngine();
void shutdownRandGenEngine();
void rewindRandList(int);
double randNum();
void expandRandList(int);
double genrand();

static struct {
	double* randList;
	int nextRand;
	int listSize;
} randGenEngine;

void initRandGenEngine() {
	/* set the initial values of nextRand and listSize */
	randGenEngine.nextRand = 0;
	randGenEngine.listSize = 0;
	randGenEngine.randList = NULL;
	
	/* expand the randList to the starting size */
	expandRandList(MIN_LIST_SIZE);	
}

void shutdownRandGenEngine() {
	/* free the randList memory */
	free(randGenEngine.randList);
	randGenEngine.randList = NULL;
	
	/* zero out the other values */
	randGenEngine.listSize = 0;
	randGenEngine.nextRand = 0;
}

double randNum() {
	/* if there are no numbers left then expand the list by a set size */
	if(randGenEngine.nextRand >= randGenEngine.listSize)
		expandRandList(LIST_SIZE_INC);
	
	/* return the next random number */
	return(randGenEngine.randList[randGenEngine.nextRand++]);	
}

void rewindRandList(int offset) {
	/* subtract the offset */
	randGenEngine.listSize -= offset;

	/* see if we went past the beginning of the list */
	if(randGenEngine.listSize < 0)
		randGenEngine.listSize = 0;
}

void expandRandList(int expandSize) {
	int i, /* general index */
	    start = randGenEngine.listSize; /* end of the previous list */
	
	/* add the expand amount to the listSize */
	randGenEngine.listSize += expandSize;
	
	/* use realloc() allocate the extra memory and preserve the current list */
	randGenEngine.randList = (double*)realloc(randGenEngine.randList,randGenEngine.listSize * sizeof(double));
		
	/* fill the new list with values */
	for(i=start; i < randGenEngine.listSize; i++) {
		randGenEngine.randList[i] = genrand();
	}
}

/*-------------------------------------------------------------------*/
/* A C-program for TT800 : July 8th 1996 Version */
/* by M. Matsumoto, email: matumoto@math.keio.ac.jp */
/* genrand() generate one pseudorandom number with double precision */
/* which is uniformly distributed on [0,1]-interval */
/* for each call.  One may choose any initial 25 seeds */
/* except all zeros. */

/* See: ACM Transactions on Modelling and Computer Simulation, */
/* Vol. 4, No. 3, 1994, pages 254-266. */

#define NRan 25
#define MRan 7

double genrand() {
    unsigned long y;
    static int k = 0;
    static unsigned long x[NRan]={ /* initial 25 seeds, change as you wish */
                                     0x95f24dab, 0x0b685215, 0xe76ccae7, 0xaf3ec239, 0x715fad23,
                                     0x24a590ad, 0x69e4b5ef, 0xbf456141, 0x96bc1b7b, 0xa7bdf825,
                                     0xc1de75b7, 0x8858a9c9, 0x2da87693, 0xb657f9dd, 0xffdc8a9f,
                                     0x8121da71, 0x8b823ecb, 0x885d05f5, 0x4e20cd47, 0x5a9ad5d9,
                                     0x512c0c03, 0xea857ccd, 0x4cc1d30f, 0x8891a8a1, 0xa6b7aadb
                                 };
    static unsigned long mag01[2]={
                                      0x0, 0x8ebfd028 /* this is magic vector `a', don't change */
                                  };
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

#endif
