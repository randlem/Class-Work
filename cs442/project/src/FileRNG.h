#include <string>
using std::string;

#include <math.h>

#include "RawRNG.h"
#include "LocalRNG.h"
#include "BoundedBuffer.h"

#ifndef FILERNG_H
#define FILERNG_H

#define RAND_RANGE(x,y,z) ((x) + (z % ((y)-(x)+1)))

class FileRNG : public LocalRNG {
	public:
		FileRNG();
		~FileRNG();

		void seedRNG(const int);
		double genRandReal1();

		int getFileSize() const {
			return(fileSize);
		}

		string genFilename() const;

	private:
		void loadFile();
		void unloadFile();

		int seed;
		unsigned char* buffer;
		int fileSize;
};

#endif
