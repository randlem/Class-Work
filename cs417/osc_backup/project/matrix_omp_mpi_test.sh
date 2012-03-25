#PBS -N matrix
#PBS -l walltime=0:05:00
#PBS -l nodes=127:ppn=2
#PBS -j oe

mpicc -openmp matrix_omp_mpi.c -o matrix.out

for i in $(seq 63 2 127)
do
mpiexec -n $i ./matrix.out
done
