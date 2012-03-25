#include <string>
using std::string;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <fstream>
using std::ifstream;

#include "lib.h"

#ifndef __ORCHESTRA_H__
#define __ORCHESTRA_H__

typedef struct {
	int			num_sections;
	rect_t*		sections;
} orchestra_t;

orchestra_t *orchestra = NULL;

void orchestra_delete(void) {
	if (orchestra->sections != NULL) {
		delete [] orchestra->sections;
		orchestra->sections = NULL;
	}

	delete orchestra;
	orchestra = NULL;
}

void orchestra_init(string file) {
	ifstream input;

	if (orchestra != NULL)
		orchestra_delete();

	orchestra = new orchestra_t;
	memset(orchestra,0,sizeof(orchestra_t));

	input.open(file.c_str());

	input >> orchestra->num_sections;
	if (orchestra->num_sections < 0) {
		cerr << "Parse Error: Bad number of sections." << endl;
		exit(-1);
	}

	orchestra->sections = new rect_t[orchestra->num_sections];
	memset(orchestra->sections,0,sizeof(rect_t)*orchestra->num_sections);

	for(int i=0; i < orchestra->num_sections; i++) {

	}

	input.close();
}

void orchestra_draw(void) {

}



#endif
