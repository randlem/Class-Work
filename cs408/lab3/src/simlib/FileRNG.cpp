#include <fstream>
using std::ifstream;
using std::ios;

#include <iostream>
using std::cout;
using std::endl;

#include <stdlib.h>

#include "FileRNG.h"

/*******************************************
* FileRNG::FileRNG()
*
* Initialize some initial vales, and do an
* initial seed of the RNG.
*******************************************/
FileRNG::FileRNG() : buffer(NULL) {
	seedRNG(0);
}

/*******************************************
* FileRNG::~FileRNG()
*
* Unload the rng file
*******************************************/
FileRNG::~FileRNG() {
	unloadFile();
}

/*******************************************
* FileRNG::seedRNG(const int)
*
* Set the seed and load the file that the
* seed represents.
*******************************************/
void FileRNG::seedRNG(const int s) {
	// set the seed
	seed = s;

	// unload and reload the entropy file
	unloadFile();
	loadFile();
}

/*******************************************
* FileRNG::genRandReal1()
*
* Return the next random value.  Random double
* made by seeking to a random area in the file
* grabbing 64-bits (8 bytes) of data, and
* making it between 0 and 1
*******************************************/
double FileRNG::genRandReal1() {
	double d = 0.0;
	int start = RAND_RANGE(0,fileSize,(int)(LocalRNG::genRandReal1() * RAND_MAX));
	unsigned long long int rand = 0;
	int i;

	for(i=0; i < 7; ++i)
		rand = (rand << 8) | (0x000000FF & buffer[start + 1]);

	//return((double) (((double)rand / (double) 0xFFFFFFFF) / (double) 0xFFFFFFFF) * 100.0);
	return((double) ((double)rand / (double)(0xFFFFFFFFFFFFFFULL)));
}

/*******************************************
* FileRNG::loadFile()
*
* Load the file of entropy based on the seed
* value (filename).
*******************************************/
void FileRNG::loadFile() {
	ifstream file;

	// make sure we don't do anything if a file has already been loaded
	if(buffer != NULL)
		return;

	// open the input file
	file.open(genFilename().c_str(),ios::ate|ios::binary);

	// make sure the file opened
	if(!file)
		return;

	// get the file size
	file.seekg(0,ios::end);
	fileSize = file.tellg();
	file.seekg(0,ios::beg);
	fileSize -= file.tellg();

	// allocate the buffer
	buffer = new unsigned char[fileSize+1];

	// read the data into the buffer
	file.read((char*)buffer,fileSize);

	// close the input file
	file.close();
}

/*******************************************
* FileRNG::unloadFile()
*
* Unload an already loaded file of entropy
*******************************************/
void FileRNG::unloadFile() {
	// make sure there's a file to unload
	if(buffer == NULL)
		return;

	// delete the buffer memory, like mommy taught us
	delete [] buffer;
	buffer = NULL;
	fileSize = 0;
}

/*******************************************
* FileRNG::genFilename()
*
* Generate a filename for the random seed based
* on a general format where there's a directory
* prefix, a standard extension, and a filename
* based on the seed, padded to a certain lenght.
*
* This could greatly benefit from config file.
*******************************************/
string FileRNG::genFilename() const {
	string s = "";
	string ext = ".bin";
	string dir = "rand/";
	int minLength = 2;
	char b[255];

	// cheat to get a character string of the seed number
	sprintf(b,"%i",seed);

	// pad the seed number if it's not long enough
	for(int i=0; i < minLength - strlen(b); ++i)
		s += '0';

	// turn the seed number into a string
	s += b;

	// return the full path (directory + seedString + extension)
	return(dir + s + ext);
}
