#PBS -N mpi
#PBS -l walltime=0:25:00
#PBS -l nodes=15
#PBS -j oe

mpicc particle_problem_mpi.c -o particle-mpi.out

if [ $? = 0 ]
then
	echo 3 threads --------------------
#	time mpiexec -n 3 particle-mpi.out 10
#	time mpiexec -n 3 particle-mpi.out 100
	time mpiexec -n 15 particle-mpi.out 10000 > test.data
	echo -------------------------------
fi
