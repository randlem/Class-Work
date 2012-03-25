#!/bin/bash

for i in `find src/ demo/ -name *.cpp`
do
	files=$files$i" "
done

for i in `find src/ demo/ -name *.h`
do
	files=$files$i" "
done

wc -l $files