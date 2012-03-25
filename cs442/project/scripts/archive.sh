#!/bin/bash

# the base filename is todays date
filename=`date +"%Y%m%d"`

# start the count at zero
i=0

#the files in the directory to save
toarchive="CHANGELOG TODO demo docs lib makefile office.in rand scripts src xml"

# loop till we find a value of i that hasn't been used
temp=$filename-$i
while [ -e archive/$temp.tar.bz2 ]
do
	((i+=1))
	temp=$filename-$i
done

# create the filename
filename=$filename-$i

# tar up the target files
tar -cvvf $filename.tar $toarchive

# turn the tar into a bzip2 archive
bzip2 $filename.tar

# move the new archive to the archive directory
mv $filename.tar.bz2 archive

# status message
echo "Created archive/$filename.tar.bz2"