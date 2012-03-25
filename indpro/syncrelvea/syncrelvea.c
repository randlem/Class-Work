/*  parallel code using a strip geometry and time synchronization relaxtion
method: fractal model:  11. 3. 2003  Y. Shim       */

#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "random_gens.h"

#define NORTH(a) (((a - 1) >= 0) ? (a - 1) : (Np - 1))
#define SOUTH(a) (((a + 1) < Np) ? (a + 1) : 0)

/* describe the area of the world we are working with*/
double  deposition  = 1.0;    /* deposition rate */
double  diffusion   = 1.e3;   /* diffusion rate */
double  CMAX        = 1.0;    /* maximum coverage */
double  CMIN        = 1.0e-4; /* minimum coverage for data taking */
double  varcycle    = 1.0;    /* target number for variable time interval */
double  factor      = 1.0;    /* multiplication factor for time interval */
double  undofactor  = 1.143;  /* event + undo factor */

int     HKcount     = 0;      /* switch for Hoshen-Kopelmann cluster counting algorithm */
int     UDTIME      = 1;      /* times for checking updaterate/sec */
int     Cevent      = 0;      /* switch for calculating Max-Min event */
int     distflag    = 0;      /* flag for checking distribution of iteration */
int     varsw       = 0;      /* 1=use number of events as a target quantity: 0=fixed time interval */

#define ndata   100
#define dcycle  50            /* to check if data taking is needed */
#define ecycle  10            /* see av.event per every ecycle to modify tau */
#define Amax    500           /* maximum size of array for boundary update: buff[2][]*/
#define Rmax    20000         /* array size of ranlist[] */
#define Nx      256           /* x dimension for each processor (not including ghost-region) */
#define Ny      1024          /* y dimension for each processor (not including ghost-region) */
#define Np      4             /* total number of processors */
#define Nxy     (Nx*Ny)
#define Emax    10000         /* array size of myeventlist[] */
#define Umax    100000        /* array size of indexc[] & ipointc[] for the undo event */
#define tlist   5000          /* array size of nbbdylist[][tlist] and mybdylist[tlist] */
#define Lx     (Np*Nx)        /* linear dimension of system along x-direction */
#define Ly      Ny            /* linear dimension of system along y-direction */
#define Lsq    (Lx*Ly)
#define nrun    1             /* number of runs */
#define Nsq    (Nx*Ny)
#define Lsmax  Nsq
#define glmax  Lsq
#define NyNp   (Ny*Np)
#define Nxp1   (Nx+1)
#define Nxp2   (Nx+2)
#define NDIR     4            /* number of directions */
#define NDIRM    3
#define Nyp1   (Ny+1)
#define Nyp2   (Ny+2)
#define Nym1   (Ny-1)
#define ndn6   (Nxp1*Ny*NDIR)
#define nn6    (Nxp2*Ny*NDIR)       /* fractal case: width of ghost region is ne */
#define totaldep (deposition*Nxy)   /* total deposition rate */
#define SEED   18529
#define ncov    3
#define maxi 99999999               /* for cluster counting */

int h[Nxp2][Ny], hT[Lx][Ly];
int nw, row[Lx][Ly], lptr[glmax];
int ipointa[ndn6], indexa[ndn6];
int nbxa[Nxp2][5], nbya[Ny][5], ndeposit, ideposit;
int list[ndn6], myid, procid[2], idata, nbdy0, nbdy1, tnbdyevent, ndiffuse;
int listold[ndn6];
int bdy[2], cbdy[2], ghost[2], cghost[2], nbnbdy[2], oldnbnbdy[2];
int nmonomer, nisland, nscluster[ncov][Lsmax];
int iran, nevent, subcycle, ncycle;
int undoflag, redoflag, nupdate, oldnevent;
int nwold, idepositold, nindex, nlist, nipoint, ncomm;
int maxnbdy0, maxnbdy1, nofbdy, iranmax, ndifbdy, nstvar;

/* data structure for kmc event */
typedef struct {
    int x,y,i,j,ranseq,tag,hxy,hij;
    double t;
} kmcevent;

/* data structure for boundary event received from neighbors*/
typedef struct {
    int x,y,hxy,i,j,hij;
    double t;
} bdyevent;

/* structure for boundary event sorting */
typedef struct {
    int dir,id;
    double t;
} sortevent;

/* structure for undoevent */
typedef struct {
    int old,site;
} changelist;

/* TODO: comment each one of these fscking variables */
changelist indexc[Umax], listc[Umax], ipointc[Umax];
kmcevent myeventlist[Emax], oldmyeventlist[Emax];
bdyevent nbbdylist[2][tlist], oldnbbdylist[2][tlist];
sortevent sortbdyevent[2*tlist];

double suni(), uni(), genrand(), dsuni(unsigned long k), duni();
double avh[ndata], cov, dcov, covtime, RLSQ, neteff, RNp, icevent;
double wlt[ndata], wsq[ndata], evt[ndata], utl[ndata];
double ievent, effutil, cevent[ndata], covc[ndata], covtimea[ndata];
double walkera[ndata], cova[ndata], walker2[ndata], buff[2][Amax];
double cvd[ndata], mds[ndata], nid[ndata], mds2[ndata], nid2[ndata];
double mytime, ranlist[Rmax], hav, CMAXfactor, tauvar;
double tau;
double timeinterval[ndata], maxtau, mmevent[ndata], avdif;
double distit[2*Lsmax];

FILE *outfile;
/* end of global variables */

void upnbhd(int a, int b);  /* for updating neighborhood */
void Deposit();             /* deposit a particle */
void Diffuse();             /* diffuse a particle */
void takedata();            /* do some measurements */
void delete(int index, int isite);
void add(int newindex, int isite);
void deletesave(int index, int isite);
void addsave(int newindex, int isite);

void update(int a, int b,int dir);  /* update neighborhood */
int  icount(int a, int b,int dir);  /* check neighbors to determine whether D is possible */
void fillBufferA(int h1,int i1, int j1, double evntime); /* fill boundary events in the buffer for communication */
void BufferSendRecv();              /* communication boundary buffer */
void updateBuffer(int iranflag);    /* update boundary event just received */
void calctime();                    /* calculate event time */
void save_random_number();          /* save random number */

void saveconfig();                  /* save some initial configurations */
void cluster(int iset);             /* HK cluster counting */
void undoevent();                   /* undo kmc event to restore original configuration */
void restoreconfig();               /* restore some initial configuration */
void sorting_nbevent();             /* sorting boundary events in early time order */
void compare_bdylist();             /* compare boundary event to determine whether redo is needed */
void save_eventlist();              /* save my kmc event */
void dokmc();                       /* determine type of kmc event: deposition or diffusion */
void save_bdylist();                /* save boundary events for comparison */
void dumpcal(int kmax);             /* some dummy calculaion */

double startwtime, endwtime;
double starteventtime, endeventtime;

int main(int argc,char* argv[])
{
	/* TODO: comment each one of these fscking variables */
    int long ISEED;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int  namelen, site, dir, id;
    int  i, j, a, tchange, x, y, hxy, hij, ranseq, tauflag;
    int  nutil, nsim, iranflag;
    int  numprocs, tnevent, tag, difevent;
    int  maxsubcycle, tncycle, maxnevent, maxiran, neventa, neventb;
    int  maxnindex, maxnipoint, maxnlist, maxnw;
    double tevent, avwalker, sqwalker, flwalker, dt, cov1, covtx;
    double avht, etsum, xraninit, teffutil1, avsubcycle;
    double wlta, wsqa, flwt, rnrun, rtnrun, RNSQ, tneteff1, tsubcycle;
    double updaterate, tupdaterate, dtevent, RR;
    double devent, tdt, tpdt, tneteff, teffutil, ldcov, t, tmytime;
    double md, md2, tid, tid2, erm, ern,  logCMIN, effutil;
    double avnevent;
    double eneventb, avnevent1, subnevent, dncycle, avtau, avtau1, dNx;
    double xevent, avevent, sdifevent, sdfevent, anevent, bnevent;
    double tanevent, tbnevent, sratio, sratio1, ratio, stanevent, stbnevent;
    double avratio, avratio1, avanevent, avbnevent, multi;
    double avnbdy, avnbdys, avnbdy1, avbdy, avevent1, timeint, mmeventx;
    double sndifbdy, avdifbdy, snstvar, avstvar;
    double multiold, xnevent, avmax;
    double sumratio, bnevent1, tbnevent1;
    double sumratio1, avnitnmax1, avmax1;

    MPI_Status stat;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    if (numprocs != Np) {
        printf("numprocs=%d Np=%d\n",numprocs,Np);
		MPI_Finalize();
        exit(1);
    }

    if (Nx > Ny) {
        printf("make Nx < Ny: your Nx=%d\t Np=%d\t Ny=%d\n",Nx,Np,Ny);
		MPI_Finalize();
		exit(1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(processor_name,&namelen);

    /* here 0 and 1 are directions */
    procid[0] = NORTH(myid);
    procid[1] = SOUTH(myid);

    bdy[0]    = 1;
    cbdy[0]   = Nxp1;
    bdy[1]    = Nx;
    cbdy[1]   = 0;
    ghost[0]  = 0;
    cghost[0] = Nx;
    ghost[1]  = Nxp1;
    cghost[1] = 1;

    startwtime = MPI_Wtime();

    RR = diffusion/deposition;
    if (myid == 0) {
        printf("CMAX=%g\t Lx=%d\t Ly=%d\t Nx=%d\t Ny=%d\t varcycle=%8.4f undofactor=1+%6.3f\n",CMAX,Lx,Ly,Nx,Ny,varcycle,undofactor-1);
        printf("Np=%d\t nrun=%d\t D/F=%10.3e\t F=%8.3f\t factor(T=factor/D)=%8.4f varsw=%d\n",Np,nrun,RR,deposition,factor,varsw);
        printf("dcycle=%d\t Amax=%d\t Rmax=%d\t Emax=%d\t Umax=%d\t tlist=%d\n",dcycle,Amax,Rmax,Emax,Umax,tlist);
    }
    RLSQ  = (double) Lsq;
    RNSQ  = (double) Nsq;
    RNp   = (double) Np;
    rnrun  = (double) nrun ;
    rtnrun = sqrt(rnrun-1.0);
    if (nrun == 1) {
        rtnrun=1.0;
	}
    dcov  = CMAX/ndata;
    ldcov = (log(CMAX)-log(CMIN))/ndata;
    logCMIN = log(CMIN);
    CMAXfactor = CMAX/factor;
    tau = (factor / diffusion);
    dNx = (double) Nx;
    maxtau = (dNx * dNx) / (2.0F * diffusion);

    /* initialize random number & discard some numbers */
    ISEED = (2 * SEED * (myid + 1)) + 1;
    suni(ISEED);
    for (a=0; a < 100; a++) {
        xraninit = uni();
	}
    for (a=0; a < 100; a++) {
        xraninit=genrand();
	}

    /* initialize arrays for data taking */
    if (distflag == 1) {
		memset(distit,0.0,sizeof(double) * 2 * Lsmax);
    }
    if (HKcount == 1) {
		memset(nscluster,0,sizeof(int) * Lsmax * ncov);
		memset(cvd,0.0,sizeof(double) * ndata);
		memset(mds,0.0,sizeof(double) * ndata);
		memset(mds2,0.0,sizeof(double) * ndata);
		memset(nid,0.0,sizeof(double) * ndata);
		memset(nid2,0.0,sizeof(double) * ndata);
    }

	memset(covc,0.0,sizeof(double) * ndata);
	memset(walkera,0.0,sizeof(double) * ndata);
	memset(walker2,0.0,sizeof(double) * ndata);
	memset(cova,0.0,sizeof(double) * ndata);
	memset(covtimea,0.0,sizeof(double) * ndata);
	memset(avh,0.0,sizeof(double) * ndata);
	memset(wlt,0.0,sizeof(double) * ndata);
	memset(wsq,0.0,sizeof(double) * ndata);
	memset(evt,0.0,sizeof(double) * ndata);
	memset(utl,0.0,sizeof(double) * ndata);
	memset(cevent,0.0,sizeof(double) * ndata);
	memset(timeinterval,0.0,sizeof(double) * ndata);
	memset(mmevent,0.0,sizeof(double) * ndata);

    updaterate  = 0.0;
    tdt         = 0.0;
    avtau       = 0.0;
    tneteff     = 0.0;
    teffutil    = 0.0;
    cov1        = 0.0;
    maxsubcycle = 0;
    tsubcycle   = 0.0;
    tnevent     = 0.0;
    tncycle     = 0;
    maxnevent   = 0;
    maxiran     = 0;
    ncomm       = 0;
    neteff      = 0.0;
    maxnindex   = 0;
    maxnipoint  = 0;
    maxnlist    = 0;
    maxnw       = 0;
    maxnbdy0    = 0;
    maxnbdy1    = 0;

    /* start main loop */
    /*------- nms statistical average ---------*/
    for (nsim=0; nsim < nrun; nsim++) {

        ndeposit  = 0;
        ideposit  = 0;
        ndiffuse  = 0;
        idata     = 0;
        ievent    = 0.0;
        nutil     = 0;
        subnevent = 0.0;
        sdifevent = 0.0;
        avnbdy    = 0.0;
        sndifbdy  = 0.0;
        snstvar   = 0.0;
        anevent   = 0.0;
        bnevent   = 0.0;
        bnevent1  = 0.0;
        avmax     = 0.0;
        avmax1    = 0.0;
        sumratio  = 0.0;
        sumratio1 = 0.0;

        /* initialized height array to zero */
		memset(h,0,sizeof(int) * Nxp2 * Ny);

        /* set neighbor's position depending on directions */
        for (a=0; a < Nxp2; a++) {
            nbxa[a][0] = (a + 1);
            nbxa[a][1] = a;
            nbxa[a][2] = (a - 1);
            nbxa[a][3] = a;
        }

        for (a=0; a < Ny; a++)	{
            nbya[a][0] = a;
            nbya[a][1] = (a - 1 + Ny) % Ny;
            nbya[a][2] = a;
            nbya[a][3] = (a + 1) % Ny;
        }

        if (Np == 1) {/* THIS HAS NOW BEEN FIXED */
            nbxa[Nx][0] = 1;
            nbxa[1][2]  = Nx;
        }

        /* initialize index, ipoint, list arrays */
        nw = 0;
		memset(ipointa,-1,sizeof(int) * ndn6);
		memset(indexa,-1,sizeof(int) * ndn6);
		memset(list,-1,sizeof(int) * ndn6);

        /* initialize update neighborhood */
        for (i=1; i < Nxp1; i++) {
            for (j=0; j < Ny; j++) {
                upnbhd(i,j);
            }
        }

        ncycle = 0;
        multi  = 1.0;
        tauvar = tau * multi;

        if (UDTIME != 0) {
            starteventtime = MPI_Wtime();
		}

        cov     = 0.0;
        covtime = 0.0;
        avtau1  = 0.0;
        tauflag = 0;

        /* do kmc until coverage < some desirable coverage */
        for (; cov <= CMAX;) {
            /* for variational time interval */
            if (tauflag == 1)
                tauvar = tauvar * multi;
            else
                tauvar = tauvar;
            if (tauvar > maxtau)
                tauvar = maxtau;

            avtau1 = avtau1 + tauvar;
            if (ncycle == 0)
                iranmax = Rmax;

            /* make enough random numbers for the current use and save them for future use */
            save_random_number();

            /* initialize time and increase number of cycle */
            mytime = 0.0;
            ncycle++;
            covtime = ncycle * tauvar;

            /* save initial ideposit, nw, list etc */
            saveconfig();

            nevent   = 0;
            iran     = 0;
            subcycle = 1;
            neventb  = 0;
            iranmax  = 0;

			/* initialize buffers */
			memset(buff,-1.0,sizeof(double) * 2 * Amax);

            nbdy0      = 0;
            nbdy1      = 0;
            buff[0][0] = 0.0;
            buff[1][0] = 0.0;

            /* random sequence start from generating an event time: save the starting point, i.e.,iran */
            myeventlist[nevent].ranseq = iran;

            /* calculate event time */
            calctime();
            nindex  = 0;
            nlist   = 0;
            nipoint = 0;

            /* do kmc event until mytime < T  */
            while (mytime < tauvar) {
                if (mytime < tauvar) {
                    dokmc();
                }
                myeventlist[nevent].ranseq = iran;
                calctime();
            }

            /* initialize number of boundary event from neighbors  */
            nbnbdy[0] = 0;
            nbnbdy[1] = 0;

            /*save_eventlist();*/

            /* 1st send/recv Buffer */
            if (Np != 1) {
                BufferSendRecv();
                ncomm++;

                /* save boundary event from neighbors for future comparison */
                save_bdylist();
            }

            /* calculate fluctuations in nbdy events */
            avnbdy1 = ((double) (nbnbdy[0] + nbnbdy[1])) / 2.0;

            if (nbdy0 > maxnbdy0)
                maxnbdy0 = nbdy0;
            if (nbdy1 > maxnbdy1)
                maxnbdy1 = nbdy1;
            if (nevent > maxnevent)
                maxnevent = nevent;
            if (iran > maxiran)
                maxiran = iran;
            if (nlist > maxnlist)
                maxnlist = nlist;
            if (nindex > maxnindex)
                maxnindex = nindex;
            if (nipoint > maxnipoint)
                maxnipoint = nipoint;
            if (nw > maxnw)
                maxnw = nw;
            if (nevent > 0)
                nutil++;

            neventa  = nevent;
            iranmax  = iran;
            bnevent  = bnevent + 1.0 *nevent;
            bnevent1 = bnevent1 + 1.0 *nevent;

            /* start iteration */
			do {
				/* start iteration from here */
				undoflag   = -1;
				redoflag   = 0;
				nofbdy     = 0;
				tnbdyevent = 0;
				avnbdy1    = ((double) (nbnbdy[0] + nbnbdy[1])) / 2.0;
				avnbdys    = avnbdys + avnbdy1;

				/* check whether new iteration is needed */
				if (subcycle == 1) {
					if (nbnbdy[0] + nbnbdy[1] > 0) {
						redoflag = 1;
						ndifbdy++;
						sorting_nbevent();
					}
				} else {
					compare_bdylist();
					if (redoflag == 1) {
						save_bdylist();
						if (nbnbdy[0] + nbnbdy[1] > 0)
							sorting_nbevent();
					}
				}

				/* new iteration is needed: redoflag=1 */
				if (redoflag == 1) {
					restoreconfig(); /* restore starting configuration */

					nupdate = 0;
					mytime  = 0.0;
					iran    = 0;
					nevent  = 0;

					/* initialize buffers */
					memset(buff,-1.0,sizeof(double) * 2 * Amax);

					nbdy0 = 0;
					nbdy1 = 0;
					buff[0][0] = 0.0;
					buff[1][0] = 0.0;

					myeventlist[nevent].ranseq = iran;
					calctime();

					/* save numbers of changes */
					nindex  = 0;
					nlist   = 0;
					nipoint = 0;

					/* repeat kmc event : update buffers and start from there*/
					/* mytime is later than 1st boundary event time */
					if (mytime > tauvar && tnbdyevent>0) {
						tmytime = tauvar+1.0;
						for (; tmytime > tauvar && nupdate < tnbdyevent;) {
							iranflag = 0;
							updateBuffer(iranflag);
							myeventlist[nevent].ranseq = iran;
							calctime();
							tmytime = mytime;
						}
					}

					/* mytime is earlier than 1st boundary event time */
					while (mytime < tauvar) {
						if (mytime < tauvar) {
							if (nupdate < tnbdyevent) {
								if (mytime < sortbdyevent[nupdate].t) {
									dokmc();
								} else {
									iranflag = 1;
									updateBuffer(iranflag);
								}
							} else {
								dokmc();
							}
						}
						myeventlist[nevent].ranseq = iran;
						calctime();
						tmytime = mytime;
						for (;tmytime > tauvar && nupdate < tnbdyevent;) {
							iranflag = 0;
							updateBuffer(iranflag);
							myeventlist[nevent].ranseq = iran;
							calctime();
							tmytime = mytime;
						}
					}
				}

				neventb = neventb+nevent;
				if (nlist > maxnlist)
					maxnlist = nlist;
				if (nindex > maxnindex)
					maxnindex = nindex;
				if (nipoint > maxnipoint)
					maxnipoint = nipoint;
				if (nw > maxnw)
					maxnw = nw;

				/* check how many processors have a change in the previous events */
				subcycle++; /* increase number of iteration */

				/* check how many processors were redone */
				MPI_Allreduce(&redoflag,&tchange,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

				if (tchange > 0) {
					/* some processors are unhappy: redo must be needed */
					bnevent = bnevent + redoflag * nevent;
					bnevent1 = bnevent1 + 1.0 * nevent;
					if (iran > iranmax)
						iranmax = iran;
					if (nevent > maxnevent)
						maxnevent = nevent;
					if (iran > maxiran)
						maxiran = iran;

					if (Np != 1)
						BufferSendRecv();
					ncomm++;
					if (nbdy0 > maxnbdy0)
						maxnbdy0 = nbdy0;
					if (nbdy1 > maxnbdy1)
						maxnbdy1 = nbdy1;
				} else {
					break;
				}

			} while(1);

            /* every processor is happy so move to next cycle */
            ievent  = ievent + 1.0 * nevent;
            tsubcycle = tsubcycle + 1.0 * subcycle;
            if (subcycle != 1) {
                bnevent = bnevent + 1.0 * nevent;
                bnevent1 = bnevent1 + 1.0 * nevent;
            }
            anevent = anevent + 1.0 * nevent;
            tncycle++;

            if (iran > iranmax)
                iranmax = iran;
            if (subcycle > maxsubcycle)
                maxsubcycle = subcycle;
            if (nevent > maxnevent)
                maxnevent = nevent;
            if (iran > maxiran)
                maxiran = iran;

			if (distflag == 1)
                distit[subcycle] = distit[subcycle] + 1.0;

			/* target quantity = av.event */
            tauflag=0;

			/* for variational time interval */
			if (varsw == 1 && (ncycle % ecycle) == 0) {
                tauflag = 1;
                xnevent = (double) nevent;
                MPI_Allreduce(&xnevent,&avevent1,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                avevent = avevent1 / RNp;
                multiold = multi;
                if (avevent > 0.0)
                    multi = varcycle / avevent;
                else
                multi = multiold;
            }

            neteff    = neteff + xevent;
            subnevent = subnevent + eneventb;/*no of events in the iteration part */

            /* determine coverage & do some measurement */
            if ((ncycle % dcycle) == 0) {
                MPI_Allreduce(&ideposit,&ndeposit,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
                cov = ((double) ndeposit) / RLSQ;

                if (log(cov) > logCMIN + ldcov * (idata + 1))
                    takedata();
            }
        }

        avtau1 = avtau1 / ((double)ncycle);
        avtau = avtau + avtau1;

        /* check update rate */
        if (UDTIME != 0) {
            endeventtime = MPI_Wtime();
            dtevent = endeventtime - starteventtime;
            tdt = tdt + dtevent;
            updaterate = updaterate + ievent / dtevent;
        }

        /* check performance */
        dncycle = (double) ncycle;
        avnbdy  = avnbdy / dncycle;
        tnevent = tnevent + ievent;/* initial no of events before iteration */
        effutil = ((double) nutil) / dncycle;
        neteff  = neteff / dncycle;/* sum of initial events and sub. events */
        tneteff  = tneteff  + neteff;
        teffutil = teffutil + effutil;
        subnevent= subnevent / dncycle;

		/* sum real / sum (real +fake) events */
        MPI_Reduce(&anevent,&tanevent,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&bnevent,&tbnevent,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&bnevent1,&tbnevent1,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if (myid == 0) {
            ratio     = tanevent / tbnevent;
            sratio    = sratio + ratio;
            sratio1   = sratio1 + tanevent / tbnevent1;
            stanevent = stanevent + tanevent;
            stbnevent = stbnevent + tbnevent;
        }

        cov1 = cov1 + cov;

    }

    avtau = avtau / rnrun;
    avsubcycle = tsubcycle / ((double)tncycle);
    avnevent = tnevent / ((double)tncycle);
    if (myid == 0) {
        printf("myid=%d ncycle=%d maxnevent=%d maxiran=%d ncomm=%d maxsubcycle=%d avnevent=%10.5f\t avsubcycle=%10.5f\n",myid,ncycle,maxnevent,maxiran,ncomm,maxsubcycle,avnevent,avsubcycle);
        printf("myid=%d maxnw=%d maxnlist=%d maxnindex=%d maxnipoint=%d maxnbdy0=%d maxnbdy1=%d\n",myid,maxnw,maxnlist,maxnindex,maxnipoint,maxnbdy0,maxnbdy1);
    }

    /* calculate update rate = no. of events per proc per sec */
    if (UDTIME != 0) {
        updaterate = updaterate / rnrun;
        MPI_Reduce(&ievent,&tevent,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&updaterate,&tupdaterate,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&tdt,&tpdt,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    }
    MPI_Reduce(&avnevent,&avnevent1,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&tneteff,&tneteff1,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&teffutil,&teffutil1,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    /* print out some performance results */
    if (myid == 0) {
        if (UDTIME != 0) {
            updaterate = tupdaterate / (RNp * rnrun);
            tdt = tpdt / (RNp * rnrun);
            printf("updaterate/(proc*sec)=%12.6e\t during dt=%12.6e\t with Np = %d\n",updaterate,tdt,Np);
            printf("ncycle =%d\t av time interval=%10.5e\t tevent=%15.1f\t total dt=%f for %d proc.\n\n",ncycle,avtau,tevent,tpdt,Np);
        }

        cov1      = cov1 / rnrun;
        tneteff   = tneteff1 / (RNp * rnrun);
        teffutil  = teffutil1 / (RNp * rnrun);
        avnevent  = avnevent1 / RNp;
        printf("cov=%10.6e\t event/cycle(initial)=%10.5f  event/cycle(initial & iterations) =%10.5f  utilization=%10.5f\n",cov1,avnevent,tneteff,teffutil);
        printf("av.max-min = %10.6e\t av.fluct.in bdy event number=%10.6e\t av.bdy var in space&time=%10.6e\n",sdfevent/rnrun,avdifbdy/rnrun,avstvar/rnrun);
        printf("av. actual fluctation in bdy number + var in space&time = %10.6e\n",(avdifbdy+avstvar)/rnrun);
        printf("av. bdy event =%10.6e\n\n",avbdy/rnrun);

        /* real/(real+fake) event */
        avanevent = stanevent / (rnrun * RNp);
        avbnevent = stbnevent / (rnrun * RNp);
        avratio   = sratio / rnrun;
        avratio1  = sratio1 / rnrun;
        printf("av. real event=%10.6e\t av. real+fake nevent=%10.6e\t real/(real+fake): upper bound=%10.7f\n",avanevent,avbnevent,avratio);
        printf("lower bound=%10.7f\n",avratio1);
    }

	/* covtime=ncycle*tau;*/

	if (myid == 0) {
        printf("covtime[%d]=ncycle*tau=%f\n",myid,covtime);
	}

    MPI_Barrier(MPI_COMM_WORLD);
    endwtime = MPI_Wtime();
    dt = endwtime - startwtime;
    MPI_Reduce(&dt,&etsum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    printf("dt =%f\t; myid=%d\n",endwtime-startwtime,myid);
    MPI_Barrier(MPI_COMM_WORLD);

    /*---------------- average ---------------*/
    /* print out some measurement result */
    if (myid == 0) {
        printf("idata=%d\n",idata);
        /*	printf("cov   event/cycle utilization \n");*/
        printf("av._cov   width walker err_wt err_walker cov-avht covtime\n");
        for (a=0; a < idata; a++) {
            cov   = covc[a] / rnrun;
            avht  = avh[a] / rnrun;
            wlta  = wlt[a] / rnrun;
            wsqa  = wsq[a] / rnrun;
            flwt  = sqrt(wsqa - wlta * wlta) / rtnrun;
            /*aevt  = evt[a] / rnrun;
            autl  = utl[a] / rnrun;*/
            if (Cevent != 0) {
                devent = cevent[a] / rnrun;
			}
            covtx = covtimea[a] / rnrun;

            avwalker = walkera[a] / (rnrun * RLSQ);
            sqwalker = walker2[a] / (rnrun * RLSQ * RLSQ);
            flwalker = sqrt(sqwalker - avwalker * avwalker) / rtnrun;

            printf("%12.7f\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%10.8f\t%12.7f\n",cov,wlta,avwalker,flwt,flwalker,cov-avht,covtx);
        }

        if (HKcount == 1) {
            printf(" cov	monomer_den	island_den	err_md	  err_tid\n");
            for (a=0; a < idata; a++) {
                cov  = covc[a] / rnrun;
                md   = mds[a] / rnrun;
                tid  = nid[a] / rnrun;

                md2  = mds2[a] / rnrun;
                erm  = sqrt(md2  - md * md) / rtnrun;
                tid2 = nid2[a] / rnrun;
                ern  = sqrt(tid2 - tid * tid) / rtnrun;
                timeint  = timeinterval[a] / rnrun;
                mmeventx = mmevent[a] / rnrun;
                printf("%d\t%12.7f\t%8.5e\t%8.5e\t%10.5e\t%10.5e\t%12.6e\t%10.6e\n",a,cov,md,tid,erm,ern,timeint,mmeventx);
            }
        }

        if (distflag == 1) {
            printf("iteration vs. distribution of number of iteration\n");
            for (a=1; a < maxsubcycle; a++) {
                if (distit[a] > 0.0)
                    printf("%d\t %12.8e\n",a,distit[a]/rnrun);
            }
        }

        etsum = etsum / (1.0 * Np);
        printf("\n\n");
        printf("-------------------------------\n\n");
        printf("timing result: %16.7f\n",etsum);
    }

    MPI_Finalize();

	return(0);

} /* end main  */

/* save some starting configurations */
void saveconfig() {
    int a, b;

    nwold = nw;
    idepositold = ideposit;
    for (a=0; a < nw; a++) { /* TODO: possible replacement with memcpy() */
        listold[a] = list[a];
    }
}

/* restore starting configurations */
void restoreconfig() {
    int a, b, x, y, i, j, isite;

    nw = nwold;
    undoevent();
    for (a=0; a < nw; a++) { /* TODO: possible replacement with memcpy() */
        list[a] = listold[a];
    }

    if (nindex == nipoint) {
        for (a=nindex-1; a >= 0; a--) {
            isite = ipointc[a].site;
            ipointa[isite] = ipointc[a].old;
            isite = indexc[a].site;
            indexa[isite] = indexc[a].old;
        }
    } else {
        for (a=nipoint-1; a >= 0; a--) {
            isite = ipointc[a].site;
            ipointa[isite] = ipointc[a].old;
        }
        for (a=nindex-1; a >=0; a--) {
            isite = indexc[a].site;
            indexa[isite] = indexc[a].old;
        }
    }
}

/* save all my kmc events */
void save_eventlist() {
    int a, b;

    oldnevent = nevent;
    for (a = 0; a < nevent; a++) { /* TODO: possible replacement with memcpy() */
        oldmyeventlist[a].x   = myeventlist[a].x;
        oldmyeventlist[a].y   = myeventlist[a].y;
        oldmyeventlist[a].i   = myeventlist[a].i;
        oldmyeventlist[a].j   = myeventlist[a].j;
        oldmyeventlist[a].tag = myeventlist[a].tag;
        oldmyeventlist[a].t   = myeventlist[a].t;
    }
}

/* save all boundary events for comparison */
void save_bdylist() {
    int a, b;

    for (a=0; a < 2; a++) {
        oldnbnbdy[a] = nbnbdy[a];
        for (b=0; b < nbnbdy[a]; b++) {
            oldnbbdylist[a][b].x   = nbbdylist[a][b].x;
            oldnbbdylist[a][b].y   = nbbdylist[a][b].y;
            oldnbbdylist[a][b].hxy = nbbdylist[a][b].hxy;
            oldnbbdylist[a][b].i   = nbbdylist[a][b].i;
            oldnbbdylist[a][b].j   = nbbdylist[a][b].j;
            oldnbbdylist[a][b].hij = nbbdylist[a][b].hij;
            oldnbbdylist[a][b].t   = nbbdylist[a][b].t;
		}
    }
}

/* compare old boundary events to new ones to determine
 * new iteration is needed */
void compare_bdylist() {
    int a, b, acheck, bcheck;

    acheck = 0;
    bcheck = 0;
    for (a=0; a < 2; a++) {
        if (oldnbnbdy[a] != nbnbdy[a]) {
            redoflag = 1;
            acheck   = 1;
        } else {
            for (b=0; b < nbnbdy[a]; ) {
                if (oldnbbdylist[a][b].t != nbbdylist[a][b].t) {
                    redoflag = 1;
                    bcheck   = 1;
                    b        = nbnbdy[a];
                }
                if (oldnbbdylist[a][b].x != nbbdylist[a][b].x) {
                    redoflag = 1;
                    bcheck   = 1;
                    b        = nbnbdy[a];
                }
                if (oldnbbdylist[a][b].y!=nbbdylist[a][b].y) {
                    redoflag = 1;
                    bcheck   = 1;
                    b        = nbnbdy[a];
                }
                if (oldnbbdylist[a][b].hxy!=nbbdylist[a][b].hxy) {
                    redoflag = 1;
                    bcheck   = 1;
                    b        = nbnbdy[a];
                }
                b++;
            }
        }
    }

    if (acheck == 1) {
        ndifbdy++;
	}

    if (bcheck == 1) {
        nstvar++;
	}
}

/* sorting boundary events received from neighbors in early time order */
void sorting_nbevent() {
    int a, b, nxcv, i, j, caselabel, dir, idn;
    double t;

    undoflag = 1;
    if (nbnbdy[0] > 0 && nbnbdy[1] == 0)
        caselabel = 0;
    if (nbnbdy[0] == 0 && nbnbdy[1] > 0)
        caselabel = 1;
    if (nbnbdy[0] > 0 && nbnbdy[1] > 0)
        caselabel = 2;

    switch(caselabel) {
    case 0:
        for (a=0; a < nbnbdy[0]; a++) {
            sortbdyevent[a].dir = 0;
            sortbdyevent[a].t   = nbbdylist[0][a].t;
            sortbdyevent[a].id  = a;
        }
        tnbdyevent = nbnbdy[0];
        break;
    case 1:
        for (a=0; a < nbnbdy[1]; a++) {
            sortbdyevent[a].dir = 1;
            sortbdyevent[a].t   = nbbdylist[1][a].t;
            sortbdyevent[a].id  = a;
        }
        tnbdyevent = nbnbdy[1];
        break;
    case 2:
        tnbdyevent = nbnbdy[0] + nbnbdy[1];
        nxcv = 0;

        /* sort the events in early time order */
        for (a = 0; a < nbnbdy[0]; a++) {
            sortbdyevent[nxcv].dir = 0;
            sortbdyevent[nxcv].t   = nbbdylist[0][a].t;
            sortbdyevent[nxcv].id  = a;
            nxcv++;
        }

        for (a=0; a < nbnbdy[1]; a++) {
            sortbdyevent[nxcv].dir = 1;
            sortbdyevent[nxcv].t   = nbbdylist[1][a].t;
            sortbdyevent[nxcv].id  = a;
            nxcv++;
        }

        for (j=1; j < tnbdyevent; j++) {
            t   = sortbdyevent[j].t;
            dir = sortbdyevent[j].dir;
            idn = sortbdyevent[j].id;

            i = j - 1;
            while (i >= 0 && sortbdyevent[i].t > t) {
                sortbdyevent[i+1].t   = sortbdyevent[i].t;
                sortbdyevent[i+1].dir = sortbdyevent[i].dir;
                sortbdyevent[i+1].id  = sortbdyevent[i].id;
                i--;
            }
            sortbdyevent[i+1].t   = t;
            sortbdyevent[i+1].dir = dir;
            sortbdyevent[i+1].id  = idn;
        }
        break;
    default:
        break;
    }
}

/* generate enough random numbers and save them */
void save_random_number() {
    int a;

    for (a=0; a < iranmax; a++) {
        ranlist[a] = uni();
    }
}

/* do kmc event: either deposition or diffusion */
void dokmc() {
    double Drate, Trate, prob, rann;

    Drate = 0.25 * nw * diffusion;
    Trate = totaldep + Drate;
    prob  = Drate / Trate;
    rann = ranlist[iran];
    iran++;

    if (rann < prob) {
        Diffuse();
        ndiffuse++;
    } else {
        Deposit();
        ideposit++;
    }
}

/* for dummy calculation */
void dumpcal(int kmax) {
    int a, b, k;
    double x, xm;

    xm = 0.0;
    for (k=0; k < kmax; k++) {
        for (a=0; a < 250; a++) {
            for (b=0; b < 250; b++) {
                xm = xm + 1.0;
            }
        }
    }
}

/* calculate event time */
void calctime() {
    double Drate, Trate, dt, rann;

    Drate = 0.25 * nw * diffusion;
    Trate = totaldep + Drate;
    rann  = ranlist[iran];
    iran++;

    dt = - log(rann) / Trate;
    mytime = mytime + dt;
}

/* fill the boundary events into buffer for communication */
void fillBufferA(int h1,int i1, int j1, double evntime) {
    int a, b;

    if (i1 <= 1) {
        a = 0;
	}

    if (i1 >= Nx) {
        a = 1;
	}

    if (a == 0) {
        if (i1 == bdy[a])   {
            nbdy0++;
            buff[a][nbdy0] = (double) cbdy[a];
            nbdy0++;
            buff[a][nbdy0] = (double) j1;
            nbdy0++;
            buff[a][nbdy0] = (double) h1;
            nbdy0++;
            buff[a][nbdy0] = evntime;
        }

        if (i1 == ghost[a]) {
            nbdy0++;
            buff[a][nbdy0] = (double) cghost[a];
            nbdy0++;
            buff[a][nbdy0] = (double) j1;
            nbdy0++;
            buff[a][nbdy0] = (double) h1;
            nbdy0++;
            buff[a][nbdy0] = evntime;
        }

        buff[a][0] = (double) nbdy0;
    }
    if (a == 1) {
        if (i1 == bdy[a])   {
            nbdy1++;
            buff[a][nbdy1] = (double) cbdy[a];
            nbdy1++;
            buff[a][nbdy1] = (double) j1;
            nbdy1++;
            buff[a][nbdy1] = (double) h1;
            nbdy1++;
            buff[a][nbdy1] = evntime;
        }

        if (i1 == ghost[a]) {
            nbdy1++;
            buff[a][nbdy1] = (double) cghost[a];
            nbdy1++;
            buff[a][nbdy1] = (double) j1;
            nbdy1++;
            buff[a][nbdy1] = (double) h1;
            nbdy1++;
            buff[a][nbdy1] = evntime;
        }

        buff[a][0] = (double) nbdy1;
    }

}

/* send/received buffer to/from neighboring processors */
void BufferSendRecv() {
    int ranks, rankd, a, b, c, x, y, am1, Nbdyevent, inc, next;
    double bufs[Amax], bufr[Amax], nTrate;
    MPI_Status stat;

    for (a=0; a < 2; a++) {
        am1 = (a + 1) % 2;
        for (b=0; b < Amax; b++) {
            bufs[b] = buff[a][b];
        }
        rankd = procid[a];
        MPI_Send(bufs,Amax,MPI_DOUBLE,rankd,1,MPI_COMM_WORLD);
        ranks = procid[am1];
        MPI_Recv(bufr,Amax,MPI_DOUBLE,ranks,1,MPI_COMM_WORLD,&stat);

        Nbdyevent = (int) bufr[0];
        if (Nbdyevent > 0) {
            inc = 0;
            for (b=1; b <= Nbdyevent;) {
                nbbdylist[am1][inc].x   =  (int) bufr[b];
                nbbdylist[am1][inc].y   =  (int) bufr[b+1];
                nbbdylist[am1][inc].hxy = (int) bufr[b+2];
                nbbdylist[am1][inc].t   =  bufr[b+3];
                next = b + 7;
                if (next <= Nbdyevent) {
                    if (bufr[b+3] == bufr[next]) {
                        nbbdylist[am1][inc].i   = (int) bufr[b+4];
                        nbbdylist[am1][inc].j   = (int) bufr[b+5];
                        nbbdylist[am1][inc].hij = (int) bufr[b+6];
                        b = b + 8;
                    } else {
                        nbbdylist[am1][inc].i   = -1;
                        nbbdylist[am1][inc].j   = -1;
                        nbbdylist[am1][inc].hij = -1;
                        b=b+4;
                    }
                } else {
                    nbbdylist[am1][inc].i   = -1;
                    nbbdylist[am1][inc].j   = -1;
                    nbbdylist[am1][inc].hij = -1;
                    b = b + 4;
                }
                inc++;
                nbnbdy[am1] = inc;
            }
        } else {
            nbbdylist[am1][0].x   = -1;
            nbbdylist[am1][0].y   = -1;
            nbbdylist[am1][0].hxy = -1;
            nbbdylist[am1][0].t   = 1e30;
            nbbdylist[am1][0].i   = -1;
            nbbdylist[am1][0].j   = -1;
            nbbdylist[am1][0].hij = -1;
            nbnbdy[am1]           = 0;
        }
    }
}

/* update boundary events in proper order */
void updateBuffer(int iranflag) {
    int a, b, am1, x, y, xi, ii, abflag, mflag, sdir, dir, aid, i, j, hij, hxy;
    double newTrate, oldTrate;

    if (redoflag == 0) {
        return;
	}

    mytime = sortbdyevent[nupdate].t;

    oldTrate = 0.25 * nw * diffusion + totaldep;

    /* update boundary and ghost regions */
    sdir = sortbdyevent[nupdate].dir;
    aid  = sortbdyevent[nupdate].id;
    nupdate++;

    x = nbbdylist[sdir][aid].x;
    y = nbbdylist[sdir][aid].y;
    hxy = h[x][y];/* save old height */

    h[x][y] = nbbdylist[sdir][aid].hxy;
    xi = x;
    if (xi == 0) {
        xi=1;
	}

    if (xi==Nxp1) {
        xi=Nx;
	}

    upnbhd(xi,y);

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

        upnbhd(ii,j);

        /* determine  Diffusion direction */
        /*if (i > x) dir = 0;
        if (j < y) dir = 1;
        if (i < x) dir = 2;
        if (j > y) dir = 3;
        if (y == Nym1 && j == 0) dir = 3;
        if (y == 0 && j == Nym1) dir = 1;*/
    }

    /* add this event in my event list */
    myeventlist[nevent].x      = x;
    myeventlist[nevent].y      = y;
    myeventlist[nevent].i      = i;
    myeventlist[nevent].j      = j;
    myeventlist[nevent].t      = mytime;
    myeventlist[nevent].ranseq = iran - iranflag;
    myeventlist[nevent].tag    = 0;
    myeventlist[nevent].hxy    = hxy;
    myeventlist[nevent].hij    = hij;
    nevent++;
}

/* undo kmc events if redoflag =1 */
void undoevent() {
    int a, xi, yi, xf, yf, tag;
    double t;

    if (redoflag == 0) {
        return;
	}

    for (a=nevent-1; a >=0; a--) {
        tag  = myeventlist[a].tag;
        xi   = myeventlist[a].x;
        yi   = myeventlist[a].y;
        xf   = myeventlist[a].i;
        yf   = myeventlist[a].j;
        t    = myeventlist[a].t;

        switch (tag) {
        case 0:
            h[xi][yi] = myeventlist[a].hxy;
            if (myeventlist[a].hij != -1)
                h[xf][yf] = myeventlist[a].hij;
            break;
        case 1:
            h[xi][yi] = h[xi][yi] - 1;
            ideposit--;
            break;
        case 2:
            h[xi][yi] = h[xi][yi] + 1;
            h[xf][yf] = h[xf][yf] - 1;
            break;
        default:
            printf("Error in tag=%d\n",tag);
            MPI_Finalize();
            return; /* SHOULDN'T THIS EXIT()?!? */
        }
    }
}

/* some measurements */
void takedata() {
    double x, tneteff, teffutil, devent, mevent, Mevent;
    int a, i, j, nwtot, nnwtot;
    double psumh, sumh, widsq, pwidsq, width, md, tid, covt;
    int iset, xdisp, yloc, sbuf[Ny], rbuf[NyNp];

    covtimea[idata] = covtimea[idata] + covtime;
    if (Cevent != 0) {
        MPI_Reduce(&icevent,&mevent,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
        MPI_Reduce(&icevent,&Mevent,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        devent = Mevent - mevent;
    }

    nwtot = nw / 4;
    MPI_Reduce(&nwtot,&nnwtot,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    walkera[idata] = walkera[idata] + 1.0 * nnwtot;
    walker2[idata] = walker2[idata] + 1.0 * nnwtot * nnwtot;

    covc[idata] = covc[idata] + cov;

    timeinterval[idata] = timeinterval[idata] + tauvar;
    mmevent[idata] = mmevent[idata] + avdif;

    /* measure surface fluctuations */
    psumh = 0.0;
    for (i=1; i < Nxp1; i++) {
        for (j=0; j < Ny; j++) {
            psumh  = psumh  + (double) h[i][j];
        }
    }

    MPI_Allreduce(&psumh,&sumh,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    hav    = sumh / RLSQ;
    pwidsq = 0.0;
    for (i=1; i < Nxp1; i++) {
        for (j=0; j < Ny; j++) {
            x = h[i][j] - hav;
            pwidsq = pwidsq + x * x;
        }
    }

    MPI_Reduce(&pwidsq,&widsq,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (myid == 0) {
        width      = sqrt(widsq / RLSQ);
        avh[idata] = avh[idata] + hav;
        wlt[idata] = wlt[idata] + width;
        wsq[idata] = wsq[idata] + width * width;
    }

    /* cluster counting part */
    if (HKcount == 1) {
        for (i=0; i < Nx; i++) {
            for (j=0; j < Ny; j++) {
                sbuf[j] = h[i+1][j];
			}

            MPI_Gather(sbuf,Ny,MPI_INT,rbuf,Ny,MPI_INT,0,MPI_COMM_WORLD);
            if (myid == 0) {
                for (j=0; j < NyNp; j++) {
                    xdisp = j / Ny;
                    yloc  = j % Ny;
                    hT[i+xdisp*Nx][yloc] = rbuf[j];
                }
            }
        }
        covt = 0.0;
        if (myid == 0) {
            for (i=0; i < Lx; i++) {
                for (j=0; j < Ly; j++) {
                    if (hT[i][j] != 0)
                        covt = covt + hT[i][j];
                }
            }
            cova[idata] = cova[idata] + covt;

            iset = 0;

            /*if (inx == 24) iset = 1;
            if (inx == 49) iset = 2; */

            cluster(iset);
            md  = (1.0 * nmonomer) / RLSQ;
            tid = (1.0 * nisland ) / RLSQ;

            mds[idata]  = mds[idata]  + md;
            mds2[idata] = mds2[idata] + md * md;
            nid[idata]  = nid[idata]  + tid;
            nid2[idata] = nid2[idata] + tid*tid;
        }
    }

    idata ++;
}

/* update all neighbors */
void upnbhd(int i, int j) {
    int dir2, dir3, ii, jj;

    if ((i >= 1) && (i <= Nx)) {
        for (dir2=0; dir2 < NDIR; dir2++) {
            update(i,j,dir2);
		}
    }

    for (dir2=0; dir2 < NDIR; dir2++) {
        ii = nbxa[i][dir2];
        jj = nbya[j][dir2];
        if (ii >= 1 && ii <= Nx) {
            for (dir3=0; dir3 < NDIR; dir3++) {
                update(ii,jj,dir3);
            }
        }
    }

}/* end of upnbhd */

/* update neighbor along some specific direction */
void update(int a, int b,int dir) {
    int a1, isite, index, newindex, ni;

    isite = NDIR * (a * Ny + b) + dir; /* 4(Nx*Ny+Ny-1) + 3 < 4(Nx+1)Ny */
    index = indexa[isite];
    newindex = icount(a,b,dir);
    if (newindex != index) {
        if (index != -1)
            deletesave(index, isite);
        if (newindex != -1)
            addsave(newindex,isite);
    }
}/* end of update(int a, int b, int dir) */

/* delete the particle from list */
void delete(int index, int isite) {
    int ipos, endsite, endpos, oldindex;

    ipos    = ipointa[isite];
    endpos  = nw - 1;
    endsite = list[endpos];

    if (endpos != ipos) {
        list[ipos]       = endsite;
        ipointa[endsite] = ipos;
    }

    indexa[isite] = -1;
    nw = nw - 1;
}

/* add the particle in the list */
void add(int index, int isite) {
    list[nw]       = isite;
    indexa[isite]  = index;
    ipointa[isite] = nw;
    nw             = nw + 1;
}

/* delete the particle from list: more efficient */
void deletesave(int index, int isite) {
    int ipos, endsite, endpos, oldindex;

    ipos    = ipointa[isite];
    endpos  = nw - 1;
    endsite = list[endpos];

    if (endpos != ipos) {
        ipointc[nipoint].old  = ipointa[endsite];
        ipointc[nipoint].site = endsite;
        nipoint++;

        list[ipos]       = endsite;
        ipointa[endsite] = ipos;
    }

    indexc[nindex].old  = indexa[isite];
    indexc[nindex].site = isite;
    nindex++;

    indexa[isite] = -1;
    nw            = nw - 1;
}

/* add the particle in the list: more efficient */
void addsave(int index, int isite) {

    ipointc[nipoint].old  = ipointa[isite];
    ipointc[nipoint].site = isite;
    nipoint++;

    list[nw]       = isite;
    ipointa[isite] = nw;

    indexc[nindex].old  = indexa[isite];
    indexc[nindex].site = isite;
    nindex++;

    indexa[isite] = index;
    nw=nw+1;
}

/* check neighborhood to determine whether the particle is needed to delete
 * from or add to the list */
int icount (int a1, int b1, int dir) {
    int xa, ya, dir2, nbonds;

    xa = nbxa[a1][dir];
    ya = nbya[b1][dir];

    if (h[xa][ya] > h[a1][b1])
        return(-1);

    nbonds = 0;
    for (dir2=0; dir2 < NDIR; dir2++) {
        xa = nbxa[a1][dir2];
        ya = nbya[b1][dir2];
        if(h[xa][ya] >= h[a1][b1]) {
            nbonds++;
        }
    }
    if (nbonds == 0)
        return(0);
    else
        return(-1);
}/* end of icount */

/* deposit a particle */
void Deposit() {
    int  x, y, i, j, hxyf, x0, nwloc, hxyi;
    double ranx, rany;

    ranx = ranlist[iran];
    iran++;
    rany = ranlist[iran];
    iran++;

    x = Nx * ranx+1;
    y = Ny * rany;
    if (x == Nx + 1)
        x = Nx;
    if (y == Ny)
        y = Ny - 1;

    if (x < 1 || x > Nx) {
        /*printf("deposition x=%d y=%d\n",x,y);*/
        MPI_Finalize();
        return;
    }

    hxyi    = h[x][y];
    h[x][y] = h[x][y] + 1;
    hxyf    = h[x][y];

    /* fill the buffer: boundary event */
    if (x == 1 || x == Nx) {
        fillBufferA(hxyf,x,y,mytime);
        nofbdy++;
    }

    /* add the deposition event in my event list */
    upnbhd(x,y);
    myeventlist[nevent].x   = x;
    myeventlist[nevent].y   = y;
    myeventlist[nevent].i   = -1;
    myeventlist[nevent].j   = -1;
    myeventlist[nevent].t   = mytime;
    myeventlist[nevent].tag = 1;
    nevent++;

}/* end of Deposit() */

/* diffuse a particle */
void Diffuse() {
    int i, j, hxyf, x, y, hxyi, iwalk, site, isite, dir;
    double rann;

    /* pick a particle from list randomly */
    rann = ranlist[iran];
    iran++;

    iwalk = rann * nw;
    site  = list[iwalk];
    isite = site / NDIR;
    dir   = site % NDIR;
    x     = isite / Ny;
    y     = isite % Ny;

    i = nbxa[x][dir];
    j = nbya[y][dir];

    h[x][y] = h[x][y] - 1;
    hxyi    = h[x][y];

    /* Buffer boundary update */
    if (x == 1 || x == Nx) {
        fillBufferA(hxyi,x,y,mytime);
	}

    h[i][j] = h[i][j] + 1;
    hxyf    = h[i][j];

    /* Buffer boundary or ghost update */
    if (i <= 1 || i >= Nx) {
        fillBufferA(hxyf,i,j,mytime);
	}

    if ((x == 1 || x == Nx) && (i <= 1 || i >= Nx)) {
        nofbdy++;
	}

    if ((x == 1 || x == Nx) && (i > 1 || i < Nx)) {
        nofbdy++;
	}

    if ((x > 1 || x < Nx) && (i <= 1 || i >= Nx)) {
        nofbdy++;
	}

    /* add the event in my event list */
    myeventlist[nevent].x   = x;
    myeventlist[nevent].y   = y;
    myeventlist[nevent].i   = i;
    myeventlist[nevent].j   = j;
    myeventlist[nevent].t   = mytime;
    myeventlist[nevent].tag = 2;
    nevent++;

    /* update neighborhood */
    upnbhd(i,j);
    upnbhd(x,y);

}/* end of Diffuse (...) */

/* Hoshen-Kopelmann's cluster count algorithm */
void cluster(int kset) {
    int a, b, k, am, ap, bm, bp, idir, pcase, mini, nofs, i, j, Lxm1, Lym1;
    int sn[4], Nm1, ms, la, cl, maxnsm;

    cl       = 0;
    maxnsm   = 0;
    nmonomer = 0;
    nisland  = 0;
    Lxm1     = Lx - 1;
    Lym1     = Ly - 1;

    for (a=0; a < Lx; a++) {
        for (b=0; b < Ly; b++) {
            row[a][b] = maxi;
        }
    }
    for (a=0; a < Lx; a++) {
        for (b=0; b < Ly; b++) {
            if (hT[a][b] == 0) {
                row[a][b] = maxi;
			}

            if (hT[a][b] >= 1) {
                am = a - 1;
                ap = a + 1;
                bm = b - 1;
                bp = b + 1;
                if (a == 0)
                    am = Lxm1;
                if (b == 0)
                    bm = Lym1;
                if (a == Lxm1)
                    ap = 0;
                if (b == Lym1)
                    bp = 0;

                /* sn[0]=up:sn[1]=left;sn[2]=down;sn[3]=right;*/
                sn[0] = row[am][b];
                sn[1] = row[a][bm];
                sn[2] = row[ap][b];
                sn[3] = row[a][bp];

                for (idir=0; idir < 4; idir++) {
                    if (sn[idir] != maxi && lptr[sn[idir]] < 0) {
                        ms = lptr[sn[idir]];
                        do {
                            la = -ms;
                            ms = lptr[la];
                        }  while(ms < 0);
                        lptr[sn[idir]] = -la;
                        sn[idir] = la;
                    }
                }
                /*------------find a minimum number -----------------*/
                mini = maxi;
                for (j=0; j < 4; j++) {
                    if (mini >= sn[j])
                        mini = sn[j];
                }
                /*------------site is not connected -----------------*/
                if (mini == maxi) {
                    cl++;
                    row[a][b] = cl;
                    lptr[cl] = 1;
                    if (cl > maxnsm)
                        maxnsm = cl;
                }
                /*------------site is connected ---------------------*/
                else {
                    nofs = 1;
                    if (a < Lxm1 && b < Lym1)
                        pcase = 0;
                    if (a < Lxm1 && b == Lym1)
                        pcase = 1;
                    if (a == Lxm1 && b < Lym1)
                        pcase = 2;
                    if (a == Lxm1 && b == Lym1)
                        pcase = 3;
                    switch(pcase) {
                    case 0:
                        if (sn[0] != sn[1] && sn[0] != maxi) {
                            nofs = nofs + lptr[sn[0]];
                            lptr[sn[0]] = -mini;
                        }
                        if (sn[1] != maxi) {
                            nofs = nofs + lptr[sn[1]];
                            lptr[sn[1]] = -mini;
                        }
                        break;
                    case 1:
                        if (sn[0] != maxi && sn[0] != sn[1] && sn[0] != sn[3]) {
                            nofs = nofs + lptr[sn[0]];
                            lptr[sn[0]] = -mini;
                        }
                        if (sn[1] != maxi && sn[1] != sn[3]) {
                            nofs = nofs + lptr[sn[1]];
                            lptr[sn[1]] = -mini;
                        }
                        if (sn[3] != maxi) {
                            nofs = nofs + lptr[sn[3]];
                            lptr[sn[3]] = -mini;
                        }
                        break;
                    case 2:
                        if (sn[0] != maxi && sn[0] != sn[1] && sn[0] != sn[2]) {
                            nofs = nofs + lptr[sn[0]];
                            lptr[sn[0]] = -mini;
                        }
                        if (sn[1] != maxi && sn[1] != sn[2]) {
                            nofs = nofs + lptr[sn[1]];
                            lptr[sn[1]] = -mini;
                        }
                        if (sn[2] != maxi) {
                            nofs = nofs + lptr[sn[2]];
                            lptr[sn[2]] = -mini;
                        }
                        break;
                    case 3:
                        if (sn[0] != maxi && sn[0] != sn[1] && sn[0] != sn[2] && sn[0] != sn[3]) {
                            nofs = nofs + lptr[sn[0]];
                            lptr[sn[0]] = -mini;
                        }
                        if (sn[1] != maxi && sn[1] != sn[2] && sn[1] != sn[3]) {
                            nofs = nofs + lptr[sn[1]];
                            lptr[sn[1]] = -mini;
                        }
                        if (sn[2] != maxi && sn[2] != sn[3]) {
                            nofs = nofs + lptr[sn[2]];
                            lptr[sn[2]] = -mini;
                        }
                        if (sn[3] != maxi) {
                            nofs = nofs + lptr[sn[3]];
                            lptr[sn[3]] = -mini;
                        }
                        break;
                    default:
                        printf("error !\n");
                        break;
                    }
                    row[a][b]  = mini;
                    lptr[mini] = nofs;
                } /* end of if (mini !=maxi)*/
            } /*end of the 1st if loop */
        }  /*end of b loop */
    }  /*end of a loop */

    /*------------cluster counting -----------------*/
    for (k=1; k <= maxnsm; k++) {
        if (lptr[k] == 1)
            nmonomer = nmonomer + 1;
        if (lptr[k] > 1 )
            nisland = nisland + 1;
        if (kset == 1 && lptr[k] >= 1)
            nscluster[kset][lptr[k]] = nscluster[kset][lptr[k]] + 1;
    }
    /*---------------*/
}/* the end of cluster */

