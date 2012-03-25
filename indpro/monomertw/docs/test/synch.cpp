#include "lattice.h"
extern MPIWrapper mpi;
void synch(Lattice * newlatt)
{
int ctr,iranflag;
int tchange=1;
float tmytime;
newlatt->subcycle=1;
int ctcycles=0; 
while(tchange>0 && ctcycles<20)
{
         
           ctcycles++;
           if(newlatt->myid==1)
	   {
           // cout<<"id="<<newlatt->myid<<"count cycles"<<ctcycles<<endl;
	   }
            /* start iteration  from here */
            //newlatt->undoflag   = -1;
            newlatt->redoflag   = 0;
            newlatt->tnbdyevent = 0;
            

            /* check whether new iteration is needed */
            if (newlatt->subcycle == 1) {
                if (newlatt->bdycountrec[0] + newlatt->bdycountrec[1] > 0) {
                    newlatt->redoflag = 1;
                    newlatt->sorting_nbevent();
                }
            } else {
                newlatt->comparebdylist();
                if (newlatt->redoflag == 1) {
                    newlatt->savebdylist();
		    
                    if (newlatt->bdycountrec[0] + newlatt->bdycountrec[1] > 0)
                        newlatt->sorting_nbevent();
                }
            }

            /* new iteration is needed: newlatt->redoflag=1 */
            if (newlatt->redoflag == 1) {
                newlatt->restoreLattice(); /* restore starting configuration */

                newlatt->nupdate = 0;
                newlatt->time  = 0.0;
                newlatt->iran    = 0;
                newlatt->nevent  = 0;

                
               
                newlatt->myeventlist[newlatt->nevent].ranseq = newlatt->iran;
                newlatt->calctime();

                /* save numbers of changes */
                /* repeat kmc event : update buffers and start from there*/
                /* newlatt->time is later than 1st boundary event time */
                if (newlatt->time > newlatt->T && newlatt->tnbdyevent>0) {
                    tmytime = newlatt->T+1.0;
                    for (; tmytime > newlatt->T && newlatt->nupdate < newlatt->tnbdyevent;) { 
                        iranflag = 0;
                        newlatt->updateBuffer(iranflag);
                        newlatt->myeventlist[newlatt->nevent].ranseq = newlatt->iran;
                        newlatt->calctime();
                        tmytime = newlatt->time;
			
                    }
                }

                /* newlatt->time is earlier than 1st boundary event time */
                while (newlatt->time < newlatt->T) {
                    if (newlatt->time < newlatt->T) {
                        if (newlatt->nupdate < newlatt->tnbdyevent) {
                            if (newlatt->time < newlatt->sortbdyevent[newlatt->nupdate].t) {
                                newlatt->doKMC();
                            } else {
                                iranflag = 1;
                                newlatt->updateBuffer(iranflag);
                            }
                        } else {
                            newlatt->doKMC();
                        }
                    }
                    newlatt->myeventlist[newlatt->nevent].ranseq = newlatt->iran;
                    newlatt->calctime();
                    tmytime = newlatt->time;
                    for (;tmytime > newlatt->T && newlatt->nupdate < newlatt->tnbdyevent;) {
                        iranflag = 0;
                        newlatt->updateBuffer(iranflag);
                        newlatt->myeventlist[newlatt->nevent].ranseq = newlatt->iran;
                        newlatt->calctime();
                        tmytime = newlatt->time;
                    }
		    //cout<<"I am"<<newlatt->time<<endl;
		    //cout<<"subcycle************************"<<newlatt->redoflag<<"flag**************** "<<ctr<<endl;
                }
            }
            
             /* check how many processors have a change in the previous events */
            newlatt->subcycle++; /* increase number of iteration */

			/* check how many processors were redone */
            mpi.allReduce(&newlatt->redoflag,&tchange,1,MPI_INT,MPI_SUM);
	   

        if (tchange > 0) { /* some processors are unhappy: redo must be needed */
                        newlatt->bdycountrec[left]=0;
                        newlatt->bdycountrec[right]=0;
                         sendmsgs(newlatt);      
            }
}

}


