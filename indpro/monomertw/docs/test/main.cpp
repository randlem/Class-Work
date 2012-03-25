#include "lattice.h"

MPIWrapper mpi;
int main(int argc,char * argv[])
{
	int i,ctr,myid=0;
	int totaldep;

	mpi.init(&argc,&argv);

	myid=mpi.getRank();
	Lattice newlatt;
	float cov,COV=1,T=0.001;
	newlatt.nbhr[left]=LEFT(myid);
	newlatt.nbhr[right]=RIGHT(myid);

	while(cov<=COV)
	{
	/**/
	newlatt.time=0;
	newlatt.nevent=0;
	newlatt.randgen();
	newlatt.iran=0;
	newlatt.saveconfig();

	newlatt.bdycountrec[left]=0;
	newlatt.bdycountrec[right]=0;


	while(newlatt.time<T )
		{
			if(newlatt.time<=T)
		{
		newlatt.doKMC();
		newlatt.calctime();
		}
	       
		}
	      
	newlatt.savebdylist();
	     
	sendmsgs(&newlatt);
	synch(&newlatt);
	mpi.allReduce(&totaldep,&newlatt.ndep,1,MPI_FLOAT,MPI_SUM);
	cov=(float) totaldep/(float)(2*size*size);
	cout<<"cov="<<cov<<endl;
	ctr++;
	}

	cout<<"latone*******************"<<endl;
	newlatt.p();

	mpi.shutdown();

	return 0;
} 

