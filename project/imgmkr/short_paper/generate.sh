#!/bin/bash

logfile="output.log"
bgnoise=0.1

rm $logfile
for dim in {256,512,1024}
do
	width=$dim
	height=$dim
	centerx=$[width/2]
	centery=$[height/2]
	axislong=$[centerx/2]
	axisshort=$[axislong/2]

	for perterb in {3,5,10,15,20}
	do
#		for bgnoise in {0.1}
#		do
			for rotate in {0,15,30,45,60}
			do
			filename="ellipse-paper-$dim-$rotate-$perterb-$bgnoise"
			if [ -e $filename.dat ]
			then
				rm $filename.dat
			fi

			touch $filename.dat
			echo "$filename.png $width $height" >> $filename.dat
			echo "perterb $perterb" >> $filename.dat
			echo "color 0 0 0" >> $filename.dat
			echo "fill" >> $filename.dat
			echo "color 255 255 255" >> $filename.dat
			echo "ellipse $centerx $centery $axislong $axisshort $rotate" >> $filename.dat
			echo "background $bgnoise" >> $filename.dat
			echo "end" >> $filename.dat

			echo "Generating $filename.dat..."
			../imgmkr $filename.dat >> $logfile

			done
#		done
	done
done
