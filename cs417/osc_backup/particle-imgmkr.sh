#PBS -N single
#PBS -l walltime=0:60:00
#PBS -l nodes=1
#PBS -j oe

make
make imgmkr
cd img1
echo 1 threads --------------------
time $HOME/particle.out 1000000 | $HOME/imgmkr.out
echo -------------------------------

