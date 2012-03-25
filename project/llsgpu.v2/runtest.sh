#!/bin/bash

echo $1

logfile="$1.dat"

touch $logfile
for i in {1..20}
do
	./llsgpu $1 >> $logfile
done

