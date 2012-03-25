#PBS -N omp
#PBS -l walltime=0:15:00
#PBS -l ncpus=15
#PBS -j oe

make -f makefile-omp

echo 2 threads --------------------
OMP_NUM_THREADS=2
export OMP_NUM_THREADS
time ./particle-omp.out 10000
echo -------------------------------

echo 5 threads --------------------
OMP_NUM_THREADS=5
export OMP_NUM_THREADS
time ./particle-omp.out 10000
echo -------------------------------

echo 7 threads --------------------
OMP_NUM_THREADS=7
export OMP_NUM_THREADS
time ./particle-omp.out 10000
echo -------------------------------

echo 9 threads --------------------
OMP_NUM_THREADS=9
export OMP_NUM_THREADS
time ./particle-omp.out 10000
echo -------------------------------

echo 11 threads -------------------- 
OMP_NUM_THREADS=11 
export OMP_NUM_THREADS 
time ./particle-omp.out 10000
echo -------------------------------

echo 13 threads --------------------
OMP_NUM_THREADS=13
export OMP_NUM_THREADS
time ./particle-omp.out 10000
echo -------------------------------

echo 15 threads --------------------
OMP_NUM_THREADS=15
export OMP_NUM_THREADS
time ./particle-omp.out 10000
echo -------------------------------
