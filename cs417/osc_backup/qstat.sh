#!/bin/bash

while [ 1 ]
do
	qstat | grep "bgs0190"
	sleep 120
done
