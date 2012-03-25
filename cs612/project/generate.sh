#!/bin/bash

for typ in random sortaorder revorder
do
	for count in 500 1000 2000 4000 6000 8000
	do
		echo "generator $count $count.$typ $typ"
		./generator $count $count.$typ $typ $count*0.1
	done
done
