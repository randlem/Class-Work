#ifndef LOCALRNG_H
#define LOCALRNG_H

// period parameters
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   // constant vector a
#define UPPER_MASK 0x80000000UL // most significant w-r bits
#define LOWER_MASK 0x7fffffffUL // least significant r bits

class LocalRNG
{
   public:
      LocalRNG();
      void seedRand(unsigned long);
      unsigned long genRandInt32();
      long genRandInt31();
      double genRandReal1();
      double genRandReal2();
      double genRandReal3();
      double genRandRes53();
   
   private:
      void initByArray();
      unsigned long mt[N];
      int mti;
};
#endif
