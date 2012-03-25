//class node;
#include<iostream.h>
#include <math.h>
const int size=20;
const int latsize=size*size;
const int dir=8;
const int left=0;
const int right=1;
const int update=0;
const int diffevent=1;
const int depevent=2;


class point
{
public:
	int x,y;
	point(){x=0;y=0;};
};
class Lattice;//forward declaration

class site
{
public:
	point pos;//position on lattice
//direction to move to
	void print();
	int index; //location on list 
	int h; //location on list 
	site();
};

class boundaryevent{
public:
    site oldsite;
    site newsite;
    double t;
    int tag;
} ;

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
//check if deposited/diffused monomer is next to a monomer
	bool neighborIsMonomer(point pt,site mysite);
//random list
	//float ranlist[10000];
//old monomerlist
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
	
	void savebdylist();
	int comparebdylist();
	
	void p();
	void calctime();
	void upnbhd(site);
	void checksite(site);
	void updateBuffer(int iranflag);
	
	float time,T;
	void randgen();
	float ranlist[10000];
	boundaryevent bdyevent[2][5000],bdyeventrec[2][5000],oldbdyeventrec[2][5000];
	int bdyleftcount,bdyrightcount,bdyrightcrec,bdyleftcrec,oldbdyleftcrec,oldbdyrightcrec;
	int redoflag,nevent,nupdate,iran,ndep,subcycle;
	

};



