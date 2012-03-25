#PBS -N testglobal
#PBS -l walltime=0:10:00
#PBS -l nodes=10
#PBS -j oe

cd $HOME/monomertw

#compile the code so we have a good copy
make clean
make testglobal 2>/dev/null

#run the compiled program
if [ "$?" -eq "0" ]
then
	time mpiexec -n 2 testglobal
else
	echo "Build error!"
fi

