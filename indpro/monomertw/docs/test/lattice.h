//class node;
#ifndef LATTICE_H
#define LATTICE_H
#include<iostream>
using std::cout;
using std::endl;

#include <math.h>
#include <stdlib.h>
#include "mpiwrapper.h"
#include "boundaryevent.h"

const int size=20;
const int latsize=size*size;
const int dir=8;
const int left=0;
const int right=1;
const int update=0;
const int diffevent=1;
const int depevent=2;
const int Np=2;
#define LEFT(a) (((a - 1) >= 0) ? (a - 1) : (Np - 1))
#define RIGHT(a) (((a + 1) < Np) ? (a + 1) : 0)


class kmcevent{
public:
    site oldsite;
    site newsite;
    int ranseq;
    double t;
    int tag;
};

class slist{
public:
point monomerloc[size*size];
int mcount;
int ndep;
};
struct listofchanges{
int oldsite;
point oldval;
int newsite;
point newval;
int tag;
float time;
};

class Lattice
{
private:
//lattice array
	site location[size+2][size];
//monomer list
	point monomerloc[size*size];
//diffusion and deposition rate
	float deprate,difrate;
	bool event;
//monomer count and number of depositions and random number count
	int mcount;
//directions to move
	point mdir[dir];
//random move function
	point ranmove(site);
//check if site is bound
	int getbonds(site,point * bondpt);
//return neighbors of site
	int getnbhrs(site,point * bondpt);
//delete monomer from list	
	void deletemonomer(point mysite);
//keep track of changes to list
	void addmonomerchange(int tag);
//check if deposited/diffused monomer is next to a monomer
	bool neighborIsMonomer(point pt,site mysite);
	listofchanges change[1000];
	void restorelist(float t);
public:
	Lattice();
	~Lattice();
	
	kmcevent myeventlist[10000];
	slist oldlist; 
//check neighborhood of monomer
	bool checkupdatebonds(site mysite);
	void deposit();
	void diffuse();
	void doKMC();
//print

/*Next level*/
	void addbdyevent(site,site,float,int);
/*
	void sendrecv();
	void updateLattice();
	void restoreLattice();
*/
	void restoreLattice();
	void undoevent();
	void saveconfig();
	void restorelist();
	void restorelist(float time);
	void savebdylist();
	int comparebdylist();
	
	void p();
	void calctime();
	void upnbhd(site);
	void checksite(site);
	void updateBuffer(int iranflag);
	void sorting_nbevent();
	float time,T;
	void randgen();
	float ranlist[10000];
	boundaryevent bdyevent[2][500],bdyeventrec[2][500],oldbdyeventrec[2][500],
		          sortbdyevent[500];
	int bdycount[2],oldbdycountrec[2],bdycountrec[2];
        int myid,nbhr[2],changecount; 
	int redoflag,nevent,nupdate,iran,ndep,subcycle,nK,tnbdyevent;
	

};

void synch(Lattice * newlatt);
void sendmsgs(Lattice  * newlatt);

#endif
