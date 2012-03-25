#PBS -N single
#PBS -l walltime=0:10:00
#PBS -l nodes=1
#PBS -j oe

make

echo 1 threads --------------------
time $HOME/particle.out 9999999
echo -------------------------------

