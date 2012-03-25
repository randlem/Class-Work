#include "site.h"

#ifndef BOUNDRYEVENT_H
#define BOUNDRYEVENT_H

typedef struct {
	site oldSite;
	site newSite;
	double time;
	int tag;
} boundryEvent;

#endif
