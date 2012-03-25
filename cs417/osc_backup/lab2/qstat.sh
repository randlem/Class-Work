#!/bin/bash

while [ 1 ]
do
	qstat | grep "bgs0190"
	echo
	sleep 120
done
