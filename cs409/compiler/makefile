#######################################################
#  Define the header and object files
#######################################################

FLAGS = -Wno-deprecated -g

HDRS = compiler.h          \
       compilerexception.h \
       emitter.h           \
       literal.h           \
       operand.h           \
       parser.h            \
       parsertable.h       \
       scanner.h           \
       symbol.h            \
       table.h             \
       token.h

SRCS = compiler.cpp          \
       compilerexception.cpp \
       emitter.cpp           \
       literal.cpp           \
       operand.cpp           \
       parser.cpp            \
       parsertable.cpp       \
       scanner.cpp           \
       symbol.cpp            \
       table.cpp             \
       token.cpp

OBJ1 = compiler.o scanner.o table.o

OBJ2 = $(OBJ1) emitter.o parser.o

#######################################################
#  Build the targets
#######################################################

mini:	mini.o $(OBJ2)
	${CXX} $(FLAGS) mini.o $(OBJ2) -o mini

scan:	scan.o $(OBJ1)
	${CXX} $(FLAGS) scan.o $(OBJ1) -o scan

testtable:	testtable.o table.o compiler.o
	${CXX} $(FLAGS) testtable.o table.o compiler.o -o testtable

#scan.o mini.o $(OBJ2):	$(HDRS)
#	${CXX} $(FLAGS) -c $*.cpp -o $@

scan.o: $(HDRS) scan.cpp
	${CXX} $(FLAGS) -c $*.cpp -o $@

mini.o: $(HDRS) mini.cpp
	${CXX} $(FLAGS) -c $*.cpp -o $@

compiler.o: $(HDRS) compiler.cpp
	${CXX} $(FLAGS) -c $*.cpp -o $@

emitter.o: $(HDRS) emitter.cpp
	${CXX} $(FLAGS) -c $*.cpp -o $@

parser.o: $(HDRS) parser.cpp
	${CXX} $(FLAGS) -c $*.cpp -o $@

scanner.o: $(HDRS) scanner.cpp
	${CXX} $(FLAGS) -c $*.cpp -o $@

testtable.o: $(HDRS) testtable.cpp
	${CXX} $(FLAGS) -c $*.cpp -o $@

table.o: $(HDRS) table.cpp
	${CXX} $(FLAGS) -c $*.cpp -o $@

.PHONY:	clean count
clean:
	@/usr/bin/rm -f $(OBJ2) mini.o scan.o testtable.o mini scan testtable
count:
	@/usr/bin/wc -l *.cpp *.h

