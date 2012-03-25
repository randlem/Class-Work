#!/bin/bash

for file in `ls *.dat`
do
	echo "Generating $file"
	../imgmkr $file >/dev/null
done
