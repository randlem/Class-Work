#PBS -N omp
#PBS -l walltime=0:15:00
#PBS -l ncpus=3
#PBS -j oe

make -f makefile-omp

echo 3 threads --------------------
OMP_NUM_THREADS=3
export OMP_NUM_THREADS
time ./particle-omp.out 10 > omp-output0.data
time ./particle-omp.out 100 > omp-output1.data
time ./particle-omp.out 1000 > omp-output2.data
echo -------------------------------


