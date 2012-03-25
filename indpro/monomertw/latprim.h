#ifndef LATPRIM_H
#define LATPRIM_H

typedef struct {
	int x;
	int y;
} point;

typedef struct {
	point p;
	int h;
	int listIndex;
} site;

typedef point* pointPTR; // pointer to a point
typedef site* sitePTR;   // pointer to a site

#endif
