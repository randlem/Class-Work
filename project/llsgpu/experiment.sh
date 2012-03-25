#!/bin/bash

rm experiment/512/*.out.png
rm experiment/512/*.png.dat

for f in `ls experiment/512/*.png`
do
	./runtest.sh $f
done

