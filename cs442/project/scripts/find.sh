#!/bin/bash

for i in `find $1`
do
	if [ -f $i ]
	then
		echo $i
		grep $2 $i
		echo
	fi
done
