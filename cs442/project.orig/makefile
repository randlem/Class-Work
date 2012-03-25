#############################################
#  makefile by Ryan Scott based on code
#  provided by Dr. Ron Lancaster
#############################################

##############################################
# define the header and object files
##############################################

FLAGS = -Wno-deprecated

SIMPLUS_HEADERS =        \
	    SimPlus.h        \
	    SampST.h         \
	    Event.h          \
	    EventPool.h      \
	    EventList.h      \
	    EventHeap.h      \
	    Entity.h         \
	    EntityQueue.h    \
	    ServerEntity.h   \
	    LocalRNG.h       \
	    NormalRNG.h      \
	    ExponentialRNG.h \
	    UniformRNG.h     \
	    RawRNG.h         \
	    NetRNG.h         \
	    BoundedBuffer.h  \
	    HTTPSocket.h

OFFICE_HEADERS =         \
	    Nurse.h          \
	    Doctor.h         \
	    Administrator.h  \
	    Patient.h        \
	    EntryNode.h

OBJ1 = SimPlus.o SampST.o Event.o EventPool.o Entity.o EntityQueue.o      \
	   ServerEntity.o EventHeap.o LocalRNG.o NormalRNG.o ExponentialRNG.o \
	   UniformRNG.o HTTPSocket.o NetRNG.o RawRNG.o

OBJ2 = $(OBJ1) Nurse.o Doctor.o Administrator.o Patient.o EntryNode.o

BASELIB = -lpthread

ifeq ($(shell uname),SunOS)
	LIBS = $(BASELIB) -lxnet
else
	LIBS = $(BASELIB)
endif

##############################################
# build the targets
##############################################

simplus:    $(OBJ1) dummy.o
			$(CXX) $(CXXFLAGS) $(LIBS) dummy.o $(OBJ1) -o dummy

demo:       $(OBJ2) office.o
			$(CXX) $(CXXFLAGS) $(LIBS) office.o $(OBJ2) -o office

prompt:     $(OBJ2) prompt.o
			$(CXX) $(CXXFLAGS) $(LIBS) prompt.o $(OBJ2) -o prompt

##############################################
# dependencies
##############################################
dummy.o $(OBJ1)         : $(SIMPLUS_HEADERS)
$(OBJ2)                 : $(OFFICE_HEADERS)
office.o prompt.o       : $(OBJ2)

##############################################
# other operations
##############################################

.PHONY: clean count run
clean:
	@rm -f *.o office dummy prompt *.exe

count:
	wc -l *.h *.cpp

run:
	./office

