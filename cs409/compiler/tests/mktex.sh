for file in *.mini
do
	echo "\begin{program}"
	echo "\begin{verbatim}"
	cat $file
	echo "\end{verbatim}"
	echo "\caption{Listing of program "$file"}"
	echo "\end{program}"
	echo
done
