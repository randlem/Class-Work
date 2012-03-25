#############################################
#  makefile by Ryan Scott based on code
#  provided by Dr. Ron Lancaster
#############################################

##############################################
# define the header and object files
##############################################

INCLUDE_PATH = src

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

LIB_OBJ = SimPlus.o SampST.o Event.o EventPool.o Entity.o EntityQueue.o      \
	   ServerEntity.o EventHeap.o LocalRNG.o NormalRNG.o ExponentialRNG.o \
	   UniformRNG.o HTTPSocket.o NetRNG.o RawRNG.o

OFFICE_OBJ = Nurse.o Doctor.o Administrator.o Patient.o EntryNode.o

BASELIB = -lpthread

ifeq ($(shell uname),SunOS)
	LIBS = $(BASELIB) -lxnet
else
	LIBS = $(BASELIB)
endif

##############################################
# build the targets
##############################################

simplus:    $(LIB_OBJ) dummy.o
			$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) dummy.o $(LIB_OBJ) -o dummy

demo:       $(OFFICE_OBJ) $(LIB_OBJ) office.o
			$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) office.o $(OFFICE_OBJ) $(LIB_OBJ) -o office

prompt:     $(OFFICE_OBJ) $(LIB_OBJ) prompt.o
			$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) $(LIBS) prompt.o $(OFFICE_OBJ) $(LIB_OBJ) -o prompt

##############################################
# dependencies
##############################################
dummy.o:
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c demo/dummy/$*.cpp -o $@

$(LIB_OBJ):
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c src/$*.cpp -o $@

$(OFFICE_OBJ): $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c demo/office/$*.cpp -o $@

office.o prompt.o: $(OFFICE_OBJ)
	$(CXX) $(CXXFLAGS) $(FLAGS) -I $(INCLUDE_PATH) -c demo/office/$*.cpp -o $@

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

