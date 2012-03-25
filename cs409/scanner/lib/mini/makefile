#######################################################
#  Define the header and object files
#######################################################

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

OBJ1 = compiler.o scanner.o table.o

OBJ2 = $(OBJ1) emitter.o parser.o

#######################################################
#  Build the targets
#######################################################

mini:	mini.o $(OBJ2)
	${CXX} ${CXXFLAGS} mini.o $(OBJ2) -o mini

scan:	scan.o $(OBJ1)
	${CXX} ${CXXFLAGS} scan.o $(OBJ1) -o scan

testtable:	testtable.o table.o compiler.o
	${CXX} ${CXXFLAGS} testtable.o table.o compiler.o -o testtable

scan.o mini.o $(OBJ1) $(OBJ2):	$(HDRS)

.PHONY:	clean count
clean:
	@/usr/bin/rm -f $(OBJ2) mini.o scan.o testtable.o mini scan testtable
count:
	@/usr/bin/wc -l *.cpp *.h

