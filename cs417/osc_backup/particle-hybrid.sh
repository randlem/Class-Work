#PBS -N hybrid
#PBS -l walltime=0:60:00
#PBS -l nodes=31:ppn=2
#PBS -j oe

mpicc -openmp particle_problem_hybrid.c -o particle-hybrid.out

for i in `seq 3 2 31`; do
	echo $i threads --------------------
	time mpiexec -n $i particle-hybrid.out 9999999
	echo -------------------------------
done

