#PBS -N lab2-static
#PBS -l walltime=2:00:00
#PBS -l nodes=33
#PBS -j oe

#compile the code so we have a good copy
mpicc mandel_static.c /usr/lib/libpng.a /usr/lib/libz.so -o mandel_static

#run the compiled program
echo 5 procs
time mpiexec -n 5 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_5.png
echo 7 procs
time mpiexec -n 7 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_7.png
echo 9 procs
time mpiexec -n 9 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_9.png
echo 11 procs
time mpiexec -n 11 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_11.png
echo 13 procs
time mpiexec -n 13 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_13.png
echo 15 procs
time mpiexec -n 15 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_15.png
echo 17 procs
time mpiexec -n 17 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_17.png
echo 19 procs
time mpiexec -n 19 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_19.png
echo 21 procs
time mpiexec -n 21 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_21.png
echo 23 procs
time mpiexec -n 23 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_23.png
echo 25 procs
time mpiexec -n 25 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_25.png
echo 27 procs
time mpiexec -n 27 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_27.png
echo 29 procs
time mpiexec -n 29 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_29.png
echo 31 procs
time mpiexec -n 31 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_31.png
echo 33 procs
time mpiexec -n 33 $HOME/mandel_static -2 1 -1.3 1.3 100000 static_out_33.png


