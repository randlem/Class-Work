#PBS -N cs680lab1
#PBS -l walltime=0:30:00
#PBS -l nodes=4
#PBS -j oe

cd $HOME/monomertw

#compile the code so we have a good copy
#make clean
#make &>/dev/null
make

#run the compiled program
if [ "$?" -eq "0" ]
then
		mkdir $PBS_JOBID
		cd $PBS_JOBID
		time mpiexec ../lab1
else
	echo "Build error!"
fi

