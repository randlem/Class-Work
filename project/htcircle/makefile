htcircle: main.cpp imgutil.h gfxutil.h util.h
	g++ -g -o htcircle main.cpp -lpng -lglut

testing: testing.cpp imgutil.h gfxutil.h util.h
	g++ -g -o htctesting testing.cpp -lpng -lglut -lrt

clean:
	rm htcircle htctesting

run:
	./htcircle

runtest:
	./htctesting single_noise_1pct

debug:
	gdb ./htcircle
