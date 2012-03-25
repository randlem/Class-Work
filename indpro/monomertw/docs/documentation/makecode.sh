#!/bin/bash

if [ ! -d $1 ]
then
	echo $i is not a directory
fi

if [ -d $2 ]
then
	rm -rf $2
fi

echo "Found CPP files:"
for i in `find $1 -maxdepth 1 -name *.cpp -printf "%f "`
do
	echo "  $i"
	files=$files$i" "
done

echo
echo "Found H files:"
for i in `find $1 -maxdepth 1 -name *.h -printf "%f "`
do
	echo "  $i"
	files=$files$i" "
done

mkdir $2

for i in $files
do
	cpp2latex -s 0 $1$i >> $2/$i.tex
#	echo $1$i
done
