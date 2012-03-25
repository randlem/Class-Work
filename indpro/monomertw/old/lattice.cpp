#include "lattice.h"
#include <stdlib.h>

site::site() {
 pos.x=0;
 pos.y=0;
 index=-1;
 h=0;
 int i,j;
}

point Lattice::ranmove(site mysite) {
 point pt;
 pt.x=0;pt.y=0;

 ////cout<<xdir<<"  ";

 int prob=rand()%4;

 pt.x=mysite.pos.x+mdir[prob].x;
 pt.y=mysite.pos.y+mdir[prob].y;

 //Allow boundary movement
 //No!!!

 if(pt.x>=size+2) pt.x-=2;
 if(pt.x<0)pt.x+=2;

 if(pt.y>=size)pt.y-=2;
 if(pt.y<0)pt.y+=2;

  return pt;
}

void site::print() {
//cout<<".pos.x="<<pos.x<<endl
    //<<"y="<<pos.y<<endl;
    //<<"loc="<<loc<<endl;
}

Lattice::Lattice() {

 float ratio=0.0;
 float prob;
 point newsite;
 mcount=0;
 deprate=1.0f,difrate=1.0e3;
 ndep=0;
 bdyleftcount=0;
 bdyrightcount=0;
 nevent=0;
 time=0;
 iran=0;
 subcycle=0;
 T=0.001;

 int i=0,j=0;

 for (i=0;i<size+2;i++)
 {
   for (j=0;j<size;j++)
      {
        newsite.x=i;
        newsite.y=j;
        location[i][j].pos=newsite;
        location[i][j].h=0;
     }
  }

 //generate random numbers

 //randgen();

 /*
 generate points in the form e.g (1,-1) move one unit left in x and and one unit up
 do not allow particle to remain (0,0)
 */


/*Set directions*/



mdir[0].y= 1;  mdir[0].x= 0;
mdir[1].y= 0;  mdir[1].x= 1;
mdir[2].y=-1;  mdir[2].x= 0;
mdir[3].y= 0;  mdir[3].x=-1;
mdir[4].y= 1;  mdir[4].x= 1;
mdir[5].y= 1;  mdir[5].x=-1;
mdir[6].y=-1;  mdir[6].x= 1;
mdir[7].y=-1;  mdir[7].x=-1;

}

void Lattice::calctime() {
	float   monomer=(float)mcount,
		Drate=difrate*monomer*0.25f,
		N=(float)latsize,dt,prob,
		totaldep=deprate*N;

	prob = ranlist[iran];
        iran++;
	dt=-log(prob)/(Drate+totaldep);
	time+=dt;
}

Lattice::~Lattice() {
//delete occupiedsite;
//deletemonomerloc;
}

int Lattice::getbonds(site mysite,point * bondpt) {
int ctr=0,i=0;
point pt;
pt=mysite.pos;
for(;i<dir;i++)
{
   pt.x+=mdir[i].x;
   pt.y+=mdir[i].y;
   if((pt.x<0)
      ||(pt.x>=size+2)
      ||(pt.y<0)
      ||(pt.y>=size))
  {

  }
  else
  {
   	if(location[pt.x][pt.y].h>=mysite.h)
   	{
   		bondpt[ctr]=pt;
		ctr++;
   	}
   }
   pt=mysite.pos;
}
return ctr;
}

int Lattice::getnbhrs(site mysite,point * bondpt) {
int ctr=0,i=0;
point pt;
pt=mysite.pos;
for(;i<dir;i++)
{
   pt.x+=mdir[i].x;
   pt.y+=mdir[i].y;
   if((pt.x<0)
      ||(pt.x>=size+2)
      ||(pt.y<0)
      ||(pt.y>=size))
  {

  }
  else
  {
   	bondpt[ctr]=pt;
	ctr++;

   }
   pt=mysite.pos;
}
return ctr;
}

void Lattice::deletemonomer(point pos) {
	point lastpt;
	//dooo not delete
	if(mcount>=1)
	{
		lastpt=monomerloc[mcount-1];
		location[lastpt.x][lastpt.y].index=location[pos.x][pos.y].index;
		monomerloc[location[pos.x][pos.y].index]=monomerloc[mcount-1];
		//cout<<"delete myself"<<endl;
		location[pos.x][pos.y].index=-1;
		mcount--;
	}
	else
	{
		location[pos.x][pos.y].index=-1;
		mcount--;
	}
}

bool Lattice::neighborIsMonomer(point pt,site mysite) {
 bool bMonomer=false;
 //if index is not -1 and same height as my location
 if (location[pt.x][pt.y].index!=-1 && (location[pt.x][pt.y].h==location[mysite.pos.x][mysite.pos.y].h))
     {
        bMonomer=true;
     }
 return bMonomer;
}

bool Lattice::checkupdatebonds(site mysite) {
/*
Possible Scenarios
Deposit
1. Monomer encounters no cluster or other neighbor monomer
-No bonds
2. Monomer encounters cluster
-bond delete monomer from list
3. Monomer encounters single monomer
-bond. delete BOTH from list

Diffusion
1. Monomer encounters no cluster or other neighbor monomer
-No bonds
2. Monomer encounters cluster
-bond delete monomer from list
3. Monomer encounters single monomer
-bond. delete BOTH from list
*/
bool bond=false;
point pt, bondpt[dir],lastpt;
int i=0,j=0,ctr=0;
ctr=getbonds(mysite,bondpt);
for(i=0;i<ctr;i++)
{
//cout<<"bondpt[i].x"<<bondpt[i].x<<"bondpt[i].y"<<bondpt[i].y<<endl;
}
if(ctr==0)
{
//No cluster or monomer;
//cout<<"bado!"<<ctr;
  bond=false;
}
else
{
//for each bond recieved
  bond=true;
   for(;j<ctr;j++)
   	{
	  pt=bondpt[j];
	  //if monomer means it has an index
	  if(neighborIsMonomer(pt,mysite))
	     {
	     //delete both you and monomer
	     //delete monomer
	     //cout<<"bondpt[j].x"<<bondpt[j].x<<"bondpt[j].y"<<bondpt[j].y<<endl;
	     deletemonomer(pt);
	     //cout<<"[lastpt.y]"<<location[lastpt.x][lastpt.y].pos.x<<"[lastpt.y]"<<location[lastpt.x][lastpt.y].pos.y<<endl;
	     //cout<<"mysite.index="<<mysite.index<<endl;
	     }
	}


        //delete your self IF you are a monomer
	 if(mysite.index!=-1)
	 {
	  //rearrange list if mcount is greater than 1
	    deletemonomer(mysite.pos);
	  }

}
return bond;
}

void Lattice::upnbhd(site mysite) {
	bool bond = false;
	point pt, bondpt[dir], lastpt;
	int x,y;
	int i = 0, j = 0, ctr = 0;

	ctr = getnbhrs(mysite,bondpt);

	checksite(mysite);
	for(i=0; i < ctr; i++) {
		x = bondpt[i].x;
		y = bondpt[i].y;
		checksite(location[x][y]);
		//bond=checkupdatebonds(location[bondpt[i].x][bondpt[i].y]);
	}
}

void Lattice::checksite(site mysite) {
	bool bond = false;
	bond = checkupdatebonds(location[mysite.pos.x][mysite.pos.y]);

	if(bond == false) {
		if((location[mysite.pos.x][mysite.pos.y].index == -1) && (location[mysite.pos.x][mysite.pos.y].h > 0)) {
			location[mysite.pos.x][mysite.pos.y].index = mcount;
			monomerloc[mcount] = location[mysite.pos.x][mysite.pos.y].pos;
			mcount++;
		}
	}
}

void Lattice::saveconfig() {
  int j;
  for (j=0;j<mcount;j++)
  {
  	oldlist.monomerloc[j]=monomerloc[j];
  }

  oldlist.mcount=mcount;
  oldlist.ndep=ndep;
}

void Lattice::restorelist() {
  int j;
  for (j=0;j<oldlist.mcount;j++)
  {
  	monomerloc[j]=oldlist.monomerloc[j];
  }

  mcount=oldlist.mcount;
  ndep=oldlist.ndep;
}

void Lattice::restoreLattice() {
   undoevent();
   restorelist();
   int i,j;
   int x,y;
   /**clear lattice***/
   for(i=0;i<size+2;i++)
   {
     for(j=0;j<size;j++)
     {
        location[i][j].index=-1;
     }
   }

   /** restore indexes**/
   for(i=0;i<mcount;i++)
   {
     x=monomerloc[i].x;
     y=monomerloc[i].y;
     location[x][y].index=i;
   }
}

void Lattice::addbdyevent(site oldsite,site newsite,float,int tag) {
//add boundary events to list
if((oldsite.pos.x>=size) || (newsite.pos.x>=size))
{
	 if(oldsite.pos.x==size)
	 {
	    oldsite.pos.x=0;
	 }

	  if(oldsite.pos.x==size+1)
	 {
	    oldsite.pos.x=1;
	 }

	 if(newsite.pos.x==size)
	 {
	    newsite.pos.x=0;
	 }

	  if(newsite.pos.x==size+1)
	 {
	    newsite.pos.x=1;
	 }

	  bdyevent[right][bdyrightcount].oldsite=oldsite;
	  bdyevent[right][bdyrightcount].newsite=newsite;
	  bdyevent[right][bdyrightcount].t=time;
	  bdyevent[right][bdyrightcount].tag=tag;
	  bdyrightcount++;
}

if((oldsite.pos.x<=1) || (newsite.pos.x<=1))
{
	  if(oldsite.pos.x==0)
	 {
	    oldsite.pos.x=size;
	 }

	  if(oldsite.pos.x==1)
	 {
	    oldsite.pos.x=size+1;
	 }

	 if(newsite.pos.x==0)
	 {
	    newsite.pos.x=size;
	 }

	  if(newsite.pos.x==1)
	 {
	    newsite.pos.x=size+1;
	 }
	  bdyevent[left][bdyleftcount].oldsite=oldsite;
	  bdyevent[left][bdyleftcount].newsite=newsite;
	  bdyevent[left][bdyleftcount].t=time;
	  bdyevent[left][bdyleftcount].tag=tag;
	  bdyleftcount++;
}

}

void Lattice::deposit() {
	/*
	1. Find a location
	2. place monomer in monomer list IF NO neighbours around!
	*/
	//do not deposit on ghost region;
	int xrand=ranlist[iran]*RAND_MAX;
	iran++;
	int yrand=ranlist[iran]*RAND_MAX;
	iran++;

	int locx=xrand%(size)+1;
	int locy=yrand%(size);

	//add height
	location[locx][locy].h+=1;

	//if monomer already present transfer index to new monomer
	//check if monomer
	bool bond=false;
	//boundary event
	if((locx==1) || (locy==size)) {
		//add event to bdylist
		addbdyevent(location[locx][locy],location[locx][locy],time,depevent);
	}
	upnbhd(location[locx][locy]);
	cout<<"newdeploc.x= "<<locx<<" newdeploc.y= "<<locy<<endl;
	//add to eventlist;
	myeventlist[nevent].oldsite=location[locx][locy];
	myeventlist[nevent].newsite=location[locx][locy];
	myeventlist[nevent].ranseq=nevent;
	myeventlist[nevent].t=time;
	myeventlist[nevent].tag=depevent;
}

void Lattice::diffuse() {
	point newloc,oldloc,lastloc;
	bool bonded=false;
	float ranm=ranlist[iran];
	iran++;

	if(mcount>0) {
		int loc=(ranm)*(mcount-1);
		oldloc=monomerloc[loc];
		newloc=ranmove(location[oldloc.x][oldloc.y]);
		location[newloc.x][newloc.y].h+=1;
		location[oldloc.x][oldloc.y].h-=1;

		//boundary event
		if((newloc.x<=1 )|| (newloc.x>=size)) {
			//add event to bdylist
			//
			addbdyevent(location[oldloc.x][oldloc.y],location[newloc.x][newloc.y],time,diffevent);
		}

		if((oldloc.x<=1 )|| (oldloc.x>=size)) {
			//add event to bdylist
			addbdyevent(location[oldloc.x][oldloc.y],location[newloc.x][newloc.y],time,diffevent);
		}

		//diffusion may release trapped monomer but capture released monomer
		if(location[oldloc.x][oldloc.y].h>location[newloc.x][newloc.y].h) {
			bonded=checkupdatebonds(location[oldloc.x][oldloc.y]);
		} else {
			//Move Monomer by changing index location
			location[newloc.x][newloc.y].index=location[oldloc.x][oldloc.y].index;
			monomerloc[location[oldloc.x][oldloc.y].index]=location[newloc.x][newloc.y].pos;
			location[oldloc.x][oldloc.y].index=-1;
			bonded=checkupdatebonds(location[newloc.x][newloc.y]);
		}
	}

	//add to eventlist;
	myeventlist[nevent].oldsite=location[oldloc.x][oldloc.y];
	myeventlist[nevent].newsite=location[newloc.x][newloc.y];
	myeventlist[nevent].ranseq=nevent;
	myeventlist[nevent].t=time;
	myeventlist[nevent].tag=diffevent;
}

void Lattice::savebdylist() {
int a,b,i;
  for(i=0;i<bdyleftcrec;i++)
  {
    oldbdyeventrec[1][i]=bdyeventrec[1][i];
  }
  oldbdyleftcrec=bdyleftcrec;
}

int Lattice::comparebdylist() {
 int a, b, acheck, bcheck;

    acheck = 0;
    bcheck = 0;
    for (a=0; a < 2; a++) {
        if (oldbdyleftcrec!= bdyleftcrec) {
            redoflag = 1;
            acheck   = 1;
        } else {
            for (b=0; b < bdyleftcrec; ) {
                if (oldbdyeventrec[a][b].t != bdyeventrec[a][b].t) {
                    redoflag = 1;
                    bcheck   = 1;
                    b        = bdyleftcrec;
                }
                if (oldbdyeventrec[a][b].oldsite.pos.x != bdyeventrec[a][b].oldsite.pos.x) {
                    redoflag = 1;
                    bcheck   = 1;
                    b        = bdyleftcrec;
                }
                if (oldbdyeventrec[a][b].oldsite.pos.y!=bdyeventrec[a][b].newsite.pos.y) {
                    redoflag = 1;
                    bcheck   = 1;
                    b        = bdyleftcrec;
                }
                if (oldbdyeventrec[a][b].oldsite.h!=bdyeventrec[a][b].oldsite.h) {
                    redoflag = 1;
                    bcheck   = 1;
                    b        = bdyleftcrec;
                }
		if (oldbdyeventrec[a][b].newsite.pos.x != bdyeventrec[a][b].newsite.pos.x) {
                    redoflag = 1;
                    bcheck   = 1;
                    b        = bdyleftcrec;
                }

                if (oldbdyeventrec[a][b].newsite.pos.y!=bdyeventrec[a][b].newsite.pos.y) {
                    redoflag = 1;
                    bcheck   = 1;
                    b        = bdyleftcrec;
                }
                if (oldbdyeventrec[a][b].newsite.h!=bdyeventrec[a][b].newsite.h) {
                    redoflag = 1;
                    bcheck   = 1;
                    b        = bdyleftcrec;
                }
                b++;
            }
        }
    }
return 0;

}

void Lattice::p() {
cout<<"**************S**********************************";
float theta=0,vacancy=(float) mcount ,lat=(float)latsize;
float x;
int i;
for (i=0;i<size;i++)
{
  cout<<endl;
  for (int j=0;j<size+2;j++)
   cout<<location[j][i].h;
}

cout<<endl<<"mcount="<<mcount<<endl;
//cout<<"**************E**********************************"<<endl;
theta=(lat-vacancy)/lat;
//cout<<"Theta="<<theta<<endl;
for (i=0;i<size;i++)
{
  cout<<endl;
  for (int j=0;j<size+2;j++)
   cout<<location[j][i].index<<"   ";
}
}

void Lattice::doKMC() {
///create and save random number
float ranX;

ranX=ranlist[iran];
iran++;

float Trate,Drate;
Drate=.25*mcount*difrate;
Trate=Drate+(deprate* (float) latsize);

float prob=(Drate/Trate);

//cout<<"ranX="<<ranlist[iran]<<endl;
if(ranX<prob)
   diffuse();
else
{
  deposit();
  ndep++;
}
nevent++;

}

void Lattice::undoevent() {
    int a, xi, yi, xf, yf, tag;
    double t;

    if (redoflag == 0) {
        return;
	}

    for (a=nevent-1; a >=0; a--) {
        tag  = myeventlist[a].tag;
        xi   = myeventlist[a].oldsite.pos.x;
        yi   = myeventlist[a].oldsite.pos.y;
        xf   = myeventlist[a].newsite.pos.x;
        yf   = myeventlist[a].newsite.pos.y;
        t    = myeventlist[a].t;

        switch (tag) {
        case 0:
            location[xi][yi].h = myeventlist[a].oldsite.h;
            if (myeventlist[a].newsite.h != -1)
                location[xf][yf].h = myeventlist[a].newsite.h;
            break;
        case 1:
            location[xi][yi].h = location[xi][yi].h + 1;
            location[xf][yf].h = location[xf][yf].h - 1;
            break;
        case 2:
	    location[xi][yi].h = location[xi][yi].h - 1;
            ndep--;
            break;

        default:
            cout<<"Error in tag"<<endl;
            return; /* SHOULDN'T THIS EXIT()?!? */
        }
    }
}

void Lattice::randgen() {
int i;
for(i=0;i<10000;i++)
  {
    ranlist[i]=((float)rand()/(float)RAND_MAX);
  }
}

void Lattice::updateBuffer(int iranflag) {
    int a, b, am1, x, y, xi, ii, abflag, mflag, sdir, dir, aid, i, j, hij, hxy;
    double newTrate, oldTrate;
    point oldsite,newsite;

    x=bdyeventrec[1][nupdate].oldsite.pos.x;
    y=bdyeventrec[1][nupdate].oldsite.pos.y;

    i=bdyeventrec[1][nupdate].newsite.pos.x;
    j=bdyeventrec[1][nupdate].newsite.pos.y;

    if (redoflag == 0) {
        return;
	}
    cout<<"update buffer!"<<endl;
    cout<<"x="<<x<<" y="<<nupdate<<endl;
    cout<<"i="<<x<<" j="<<y<<endl;
    time = bdyeventrec[1][nupdate].t;

    //oldTrate = 0.25 * nw * diffusion + totaldep;

    /* update boundary and ghost regions */
    /*
    sdir = sortbdyevent[nupdate].dir;
    aid  = sortbdyevent[nupdate].id;
    nupdate++;

    x = nbbdylist[sdir][aid].x;
    y = nbbdylist[sdir][aid].y;
    hxy = h[x][y];/* save old height */

    location[x][y].h = bdyeventrec[1][nupdate].oldsite.h;
    upnbhd(bdyeventrec[1][nupdate].oldsite);
    /*
    xi = x;
    if (xi == 0) {
        xi=1;
	}

    if (xi==Nxp1) {
        xi=Nx;
	}

    upnbhd(xi,y);
     */
    /*
    dir = -1;
    i   = nbbdylist[sdir][aid].i;
    j   = nbbdylist[sdir][aid].j;
    hij = -1;
    if (i != -1 && j != -1) {
        hij = h[i][j];
        h[i][j] = nbbdylist[sdir][aid].hij;
        ii = i;
        if (ii == 0) {
            ii = 1;
		}

        if (ii == Nxp1) {
            ii = Nx;
		}
      */
    location[i][j].h = bdyeventrec[1][nupdate].newsite.h;
    upnbhd(bdyeventrec[1][nupdate].newsite);

        /* determine  Diffusion direction */
        /*if (i > x) dir = 0;
        if (j < y) dir = 1;
        if (i < x) dir = 2;
        if (j > y) dir = 3;
        if (y == Nym1 && j == 0) dir = 3;
        if (y == 0 && j == Nym1) dir = 1;*/

    /* add this event in my event list */
   myeventlist[nevent].oldsite=location[x][y];
   myeventlist[nevent].newsite=location[i][j];
   myeventlist[nevent].ranseq=iran - iranflag;
   myeventlist[nevent].t=time;
   myeventlist[nevent].tag=0;

   nupdate++;
   nevent++;
}

void sendmsgs(Lattice  * newlatt);

void synch(Lattice  * newlatt);

int main()
{
int i,ctr=0;
Lattice newlatt[2];
float cov,COV=1,T=0.001;
while(cov<=COV)
{
newlatt[0].time=0;
newlatt[1].time=0;

newlatt[0].nevent=0;
newlatt[1].nevent=0;

newlatt[0].randgen();
newlatt[1].randgen();

newlatt[0].iran=0;
newlatt[1].iran=0;

newlatt[0].saveconfig();
newlatt[1].saveconfig();



newlatt[1].bdyleftcrec=0;
newlatt[0].bdyleftcrec=0;

while(newlatt[0].time<T || newlatt[1].time<T )
    {
      for(i=0;i<2;i++)
      {
        if(newlatt[i].time<=T)
	{
	  newlatt[i].doKMC();
	  newlatt[i].calctime();
	}
      }
       //sendmsgs(newlatt);

       for(i=0;i<2;i++)
         newlatt[i].savebdylist();
    }

 for(i=0;i<2;i++)
     newlatt[i].savebdylist();

sendmsgs(newlatt);
synch(newlatt);
cov=((float) newlatt[0].ndep +(float)newlatt[1].ndep)/(float)(2*size*size);
cout<<"iran=***************************************************************************"<<newlatt[0].iran<<endl;
cout<<"iran="<<newlatt[0].iran<<endl;
cout<<"cov="<<cov<<endl;
ctr++;
}


//newlatt[0].savebdylist();
//newlatt[0].comparebdylist();
cout<<"latzero*******************"<<endl;
cout<<"redoflag="<<newlatt[0].redoflag<<endl;
newlatt[0].p();

cout<<"latone*******************"<<endl;
newlatt[1].p();

return 0;
}

void sendmsgs(Lattice  * newlatt)
{
int i,j;
//copy boundary events from left to right
newlatt[1].bdyleftcrec=newlatt[0].bdyrightcount;
cout<<"newlatt[1].bdyleftcrec*****************************************"<<newlatt[1].bdyleftcrec<<endl;
for(i=0;i<newlatt[1].bdyleftcrec;i++)
  {
    newlatt[1].bdyeventrec[1][i]=newlatt[0].bdyevent[1][i];
  }

//copy boundary events from right to left
newlatt[0].bdyleftcrec=newlatt[1].bdyleftcount;
cout<<"newlatt[0].bdyleftcrec*****************************************"<<newlatt[0].bdyleftcrec<<endl;
for(i=0;i<newlatt[0].bdyleftcrec;i++)
  {
    newlatt[0].bdyeventrec[1][i]=newlatt[1].bdyevent[0][i];
  }


  newlatt[0].bdyrightcount=0;
  newlatt[1].bdyleftcount=0;
}

void synch(Lattice * newlatt)
{
int ctr,iranflag;
int tchange=1;
float tmytime;

newlatt[0].subcycle=1;
newlatt[1].subcycle=1;

while(tchange>0)
{
for(ctr=0;ctr<2;ctr++)
{
            /* start iteration from here */
            //newlatt[ctr].undoflag   = -1;
            newlatt[ctr].redoflag   = 0;
            //newlatt[ctr].bdyleftcrec = 0;


            /* check whether new iteration is needed */
            if (newlatt[ctr].subcycle == 1) {
                if (newlatt[ctr].bdyleftcrec > 0) {
                    newlatt[ctr].redoflag = 1;
                    //sorting_nbevent(&newlatt);
                }
            } else {
                newlatt[ctr].comparebdylist();
                if (newlatt[ctr].redoflag == 1) {
                    newlatt[ctr].savebdylist();

                    //if (newlatt[ctr].nbnbdy[0] + newlatt[ctr].nbnbdy[1] > 0)
                        //sorting_nbevent(&newlatt);
                }
            }

            /* new iteration is needed: newlatt[ctr].redoflag=1 */
            if (newlatt[ctr].redoflag == 1) {
                newlatt[ctr].restoreLattice(); /* restore starting configuration */

                newlatt[ctr].nupdate = 0;
                newlatt[ctr].time  = 0.0;
                newlatt[ctr].iran    = 0;
                newlatt[ctr].nevent  = 0;



                newlatt[ctr].myeventlist[newlatt[ctr].nevent].ranseq = newlatt[ctr].iran;
                newlatt[ctr].calctime();

                /* save numbers of changes */
                /* repeat kmc event : update buffers and start from there*/
                /* newlatt[ctr].time is later than 1st boundary event time */
                if (newlatt[ctr].time > newlatt[ctr].T && newlatt[ctr].bdyleftcrec>0) {
                    tmytime = newlatt[ctr].T+1.0;
                    for (; tmytime > newlatt[ctr].T && newlatt[ctr].nupdate < newlatt[ctr].bdyleftcrec;) {
                        iranflag = 0;
                        newlatt[ctr].updateBuffer(iranflag);
                        newlatt[ctr].myeventlist[newlatt[ctr].nevent].ranseq = newlatt[ctr].iran;
                        newlatt[ctr].calctime();
                        tmytime = newlatt[ctr].time;

                    }
                }

                /* newlatt[ctr].time is earlier than 1st boundary event time */
                while (newlatt[ctr].time < newlatt[ctr].T) {
                    if (newlatt[ctr].time < newlatt[ctr].T) {
                        if (newlatt[ctr].nupdate < newlatt[ctr].bdyleftcrec) {
                            if (newlatt[ctr].time < newlatt[ctr].bdyeventrec[1][newlatt[ctr].nupdate].t) {
                                newlatt[ctr].doKMC();
                            } else {
                                iranflag = 1;
                                newlatt[ctr].updateBuffer(iranflag);
                            }
                        } else {
                            newlatt[ctr].doKMC();
                        }
                    }
                    newlatt[ctr].myeventlist[newlatt[ctr].nevent].ranseq = newlatt[ctr].iran;
                    newlatt[ctr].calctime();
                    tmytime = newlatt[ctr].time;
                    for (;tmytime > newlatt[ctr].T && newlatt[ctr].nupdate < newlatt[ctr].bdyleftcrec;) {
                        iranflag = 0;
                        newlatt[ctr].updateBuffer(iranflag);
                        newlatt[ctr].myeventlist[newlatt[ctr].nevent].ranseq = newlatt[ctr].iran;
                        newlatt[ctr].calctime();
                        tmytime = newlatt[ctr].time;
                    }
		    //cout<<"I am"<<newlatt[ctr].time<<endl;
		    //cout<<"subcycle************************"<<newlatt[ctr].redoflag<<"flag**************** "<<ctr<<endl;
                }
            }

             /* check how many processors have a change in the previous events */
            newlatt[ctr].subcycle++; /* increase number of iteration */

			/* check how many processors were redone */
            //MPI_Allreduce(&newlatt[ctr].redoflag,&tchange,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

	if(newlatt[ctr].redoflag==1)
	 {
	   tchange=1;
	 }
	else
	 {
	   tchange=0;
	 }

}

        if (tchange > 0) { /* some processors are unhappy: redo must be needed */
                         sendmsgs(newlatt);
			 newlatt[0].savebdylist();
			 newlatt[1].savebdylist();
            }
}

}


