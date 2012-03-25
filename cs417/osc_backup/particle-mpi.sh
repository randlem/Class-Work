#PBS -N mpi
#PBS -l walltime=0:30:00
#PBS -l nodes=31
#PBS -j oe

mpicc particle_problem_mpi.c -o particle-mpi.out

for i in `seq 3 2 31`; do
	echo $i threads --------------------
	time mpiexec -n $i particle-mpi.out 2000000
	echo -------------------------------
done

