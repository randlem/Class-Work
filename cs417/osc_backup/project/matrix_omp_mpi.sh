#PBS -N matrix
#PBS -l walltime=0:02:00
#PBS -l nodes=51:ppn=2
#PBS -j oe

mpicc -openmp matrix_omp_mpi.c -o matrix.out

#echo 3 nodes -----------------------
#mpiexec -n 3 ./matrix.out
#echo -------------------------------

#echo 5 nodes --------------------
#mpiexec -n 5 ./matrix.out
#echo -------------------------------

#echo 7 nodes --------------------
#mpiexec -n 7 ./matrix.out
#echo -------------------------------

#echo 9 nodes --------------------
#mpiexec -n 9 ./matrix.out
#echo -------------------------------

#echo 11 nodes --------------------
#mpiexec -n 11 ./matrix.out
#echo -------------------------------

#echo 13 nodes --------------------
#mpiexec -n 13 ./matrix.out
#echo -------------------------------

#echo 15 nodes --------------------
#mpiexec -n 15 ./matrix.out
#echo -------------------------------

#echo 17 nodes --------------------
#mpiexec -n 17 ./matrix.out
#echo -------------------------------

#echo 19 nodes --------------------
#mpiexec -n 19 ./matrix.out
#echo -------------------------------

#echo 21 nodes --------------------
#mpiexec -n 21 ./matrix.out
#echo -------------------------------

#echo 23 nodes --------------------
#mpiexec -n 23 ./matrix.out
#echo -------------------------------

#echo 25 nodes --------------------
#mpiexec -n 25 ./matrix.out
#echo -------------------------------

#echo 27 nodes --------------------
#mpiexec -n 27 ./matrix.out
#echo -------------------------------

#echo 29 nodes --------------------
#mpiexec -n 29 ./matrix.out
#echo -------------------------------

#echo 31 nodes --------------------
#mpiexec -n 31 ./matrix.out
#echo -------------------------------

#echo 33 nodes --------------------
#mpiexec -n 33 ./matrix.out
#echo -------------------------------

#echo 35 nodes --------------------
#mpiexec -n 35 ./matrix.out
#echo -------------------------------

#echo 37 nodes --------------------
#mpiexec -n 37 ./matrix.out
#echo -------------------------------

#echo 39 nodes --------------------
#mpiexec -n 39 ./matrix.out
#echo -------------------------------

#echo 41 nodes --------------------
#mpiexec -n 41 ./matrix.out
#echo -------------------------------

#echo 43 nodes --------------------
#mpiexec -n 43 ./matrix.out
#echo -------------------------------

#echo 45 nodes --------------------
#mpiexec -n 45 ./matrix.out
#echo -------------------------------

#echo 47 nodes --------------------
#mpiexec -n 47 ./matrix.out
#echo -------------------------------

echo 49 nodes --------------------
mpiexec -n 49 ./matrix.out
echo -------------------------------

echo 51 nodes --------------------
mpiexec -n 51 ./matrix.out
echo -------------------------------

#echo 53 nodes --------------------
#mpiexec -n 53 ./matrix.out
#echo -------------------------------

#echo 55 nodes --------------------
#mpiexec -n 55 ./matrix.out
#echo -------------------------------

#echo 57 nodes --------------------
#mpiexec -n 57 ./matrix.out
#echo -------------------------------

#echo 59 nodes --------------------
#mpiexec -n 59 ./matrix.out
#echo -------------------------------

#echo 61 nodes --------------------
#mpiexec -n 61 ./matrix.out
#echo -------------------------------

#echo 63 nodes --------------------
#mpiexec -n 63 ./matrix.out
#echo -------------------------------



