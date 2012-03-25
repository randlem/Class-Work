#PBS -N lab3-test
#PBS -l walltime=0:02:00
#PBS -l ncpus=17
#PBS -j oe

ecc -openmp matrix_part1.c

echo 17 threads --------------------
OMP_NUM_THREADS=17
export OMP_NUM_THREADS
time ./a.out
echo -------------------------------

