llsellipse: main.cpp imgutil.h util.h
	g++ -O2 -march=athlon64 -pipe -o llsellipse main.cpp -lpng -lglut -lrt

build-debug: main.cpp imgutil.h util.h
	g++ -g -o llsellipse main.cpp -lpng -lglut -lrt

clean:
	rm llsellipse

run:
#	./llsellipse ellipse-10.png
	./llsellipse ellipse-single-basic.png

debug:
	gdb ./llsellipse
