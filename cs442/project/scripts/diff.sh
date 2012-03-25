#!/bin/bash

cd src

for i in *
do
	diff $i ../old/$i | less
done
