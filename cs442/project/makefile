#############################################
#  makefile by Ryan Scott based on code
#  provided by Dr. Ron Lancaster
#############################################

##############################################
# define the header and object files
##############################################

INCLUDE_PATH = src

FLAGS = -Wno-deprecated -g

# a list of all the headers used in the SimPlus library
SIMPLUS_HEADERS =         \
	    SimPlus.h         \
	    SampST.h          \
	    Event.h           \
	    EventPool.h       \
	    EventList.h       \
	    EventHeap.h       \
	    Entity.h          \
	    EntityQueue.h     \
	    ServerEntity.h    \
	    LocalRNG.h        \
	    NormalDist.h      \
	    ExponentialDist.h \
	    UniformDist.h     \
	    RawRNG.h          \
	    NetRNG.h          \
	    BoundedBuffer.h   \
	    HTTPSocket.h      \
		RNGFactory.h      \
		RawDist.h         \
		PoissonDist.h     \
		BinomialDist.h    \
		BernoulliDist.h   \
		FileRNG.h         \
		LogFile.h         \
		WeibullDist.h     \
		LogLogisticDist.h \
		TriangleDist.h    \
		Image.h

OFFICE_HEADERS =         \
	    Nurse.h          \
	    Doctor.h         \
	    Administrator.h  \
	    Patient.h        \
	    EntryNode.h

MM1_HEADERS = Object.h Server.h EntryNode.h

BANK_HEADERS = Door.h Teller.h Customer.h

# a list of all the objects that will need to be built to have a fully compiled SimPlus library
LIB_OBJ = SimPlus.o SampST.o Event.o EventPool.o Entity.o EntityQueue.o         \
	   ServerEntity.o EventHeap.o LocalRNG.o NormalDist.o ExponentialDist.o     \
	   UniformDist.o HTTPSocket.o NetRNG.o RNGFactory.o RawDist.o PoissonDist.o \
	   BinomialDist.o FileRNG.o LogFile.o BernoulliDist.o WeibullDist.o         \
	   LogLogisticDist.o TriangleDist.o Image.o

# object require to build the office demo
OFFICE_OBJ = Nurse.o Doctor.o Administrator.o Patient.o EntryNode.o

# object require to build the mm1 demo
MM1_OBJ = Object.o Server.o Entry.o

# object require to build the bank demo
BANK_OBJ = Door.o Teller.o Customer.o

# the base libraries to compile SimPlu
BASELIB = -lpthread -lm -lpng

# get special libs for Soliaris (SunOS) systems
ifeq ($(shell uname),SunOS)
	LIBS = $(BASELIB) -lxnet
else
	LIBS = $(BASELIB)
endif

##############################################
# build the targets
##############################################

# compile rule to make a dummy executeable
simplus: $(LIB_OBJ) dummy.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) dummy.o $(LIB_OBJ) -o dummy

# compile rule to make the office demo
office: $(OFFICE_OBJ) $(LIB_OBJ) office.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) office.o $(OFFICE_OBJ) $(LIB_OBJ) -o office

# compile rule to make an alternate office demo
prompt: $(OFFICE_OBJ) $(LIB_OBJ) prompt.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) prompt.o $(OFFICE_OBJ) $(LIB_OBJ) -o prompt

# compile rule to make a netrng test program
netrng: $(LIB_OBJ) netrng.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) netrng.o $(LIB_OBJ) -o netrng

# compile rule to make a distribution test program
disttest: $(LIB_OBJ) disttest.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) disttest.o $(LIB_OBJ) -o disttest

# compile rule to make a filerng test program
filerng: $(LIB_OBJ) filerng.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) filerng.o $(LIB_OBJ) -o filerng

# compile rule to make a stats tester
stats: $(LIB_OBJ) stats.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) stats.o $(LIB_OBJ) -o stats

# compile rule to make a stats tester
image: $(LIB_OBJ) image.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) image.o $(LIB_OBJ) -o image

# compile rule to make a logfile tester
logfile: LogFile.o logfile.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) logfile.o LogFile.o -o logfile

# compile rule to make the mm1 demo
mm1: $(MM1_OBJ) $(LIB_OBJ) mm1.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) mm1.o $(MM1_OBJ) $(LIB_OBJ) -o mm1

# compile rule to make the bank demo
bank: $(BANK_OBJ) $(LIB_OBJ) bank.o
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) bank.o $(BANK_OBJ) $(LIB_OBJ) -o bank

##############################################
# dependencies
##############################################

# compile directive for anything in demo/dummy
%.o: demo/dummy/%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c $< -o $@

# compile directive for anything in src
%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c $< -o $@

# compile directive for anything in demo/office
%.o: demo/office/%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c $< -o $@

# compile directive for anything in demo/netrng
%.o: demo/netrng/%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c $< -o $@

# compile directive for anything in demo/disttest
%.o: demo/disttest/%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c $< -o $@

# compile directive for anything in demo/filerng
%.o: demo/filerng/%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c $< -o $@

# compile directive for anything in demo/image
%.o: demo/image/%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c $< -o $@

# compile directive for anything in demo/stats
%.o: demo/stats/%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c $< -o $@

# compile directive for anything in demo/logfile
%.o: demo/logfile/%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c $< -o $@

# compile directive for anything in demo/mm1
%.o: demo/mm1/%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c $< -o $@

# compile directive for anything in demo/bank
%.o: demo/bank/%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c $< -o $@

##############################################
# other operations
##############################################

.PHONY: clean count
# clean the object files, and any executeables present
clean:
	@rm -f *.o office dummy prompt netrng disttest filerng image stats mm1 logfile gmon.out bank

# execute the lines-of-code (loc) script; count the lines of code
count:
	scripts/loc.sh


