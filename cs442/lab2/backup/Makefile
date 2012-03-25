CC=gcc
LIB=-lm
FLG=-g

lab2: main.o simlib.o
	$(CC) $(FLG) -o lab2 main.o simlib.o $(LIB)

main.o: main.c
	$(CC) $(FLG) -o main.o -c main.c 

simlib.o: simlib.c
	$(CC) $(FLG) -o simlib.o -c simlib.c

.PHONY: clean

clean:
	@rm -rf simlib.o main.o lab2
