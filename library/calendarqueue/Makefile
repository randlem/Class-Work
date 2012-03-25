FLAGS=-Wno-deprecated -g
OBJS=cqueue.o main.o
LIBS=-lm

cqueue: $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

%.o: %.cpp
	$(CXX) -o $@ -c $<