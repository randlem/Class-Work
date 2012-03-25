#!/bin/bash

width=512
height=512
centerx=$[width/2]
centery=$[height/2]
axislong=$[centerx/2]
axisshort=$[axislong/2]

for perterb in {0..5}
do
	for noisy in {1.0,0.95,0.9,0.85,0.80}
	do
		for bgnoise in {0.0,0.01,0.02,0.05,0.1}
		do
			for rotate in {0,30,45,60}
			do
			filename="ellipse-single-$rotate-$perterb-$noisy-$bgnoise"
			if [ -e $filename.dat ]
			then
				rm $filename.dat
			fi

			touch $filename.dat
			echo "$filename.png $width $height" >> $filename.dat
			echo "perterb $perterb" >> $filename.dat
			echo "noisy $noisy" >> $filename.dat
			echo "color 0 0 0" >> $filename.dat
			echo "fill" >> $filename.dat
			echo "color 255 255 255" >> $filename.dat
			echo "ellipse $centerx $centery $axislong $axisshort $rotate" >> $filename.dat
			echo "background $bgnoise" >> $filename.dat
			echo "end" >> $filename.dat

			done
		done
	done
done
