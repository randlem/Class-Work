// BoundedBuffer.h 
//
// A templatized, thread-safe circular buffer structure
// which implements the Monitor::Notify semantic. Its size
// is a construction-time parameter.  When it is full   
// the thread(s) calling put is(are) blocked, and when
// the buffer empties the thread(s) calling get is(are)
// blocked.
//
// Requires Support of PThreads!
//
// METHODS:
// --------
//
// BoundedBuffer(unsigned short)
// Constructor; size parameter determines size of circular buffer.
// Initializes data members & pthread-related system resources.
//
// ~BoundedBuffer()
// Destructor; deallocates dynamic memory and tears down system resources.
//
// get(T& ) : bool
// A monitor function which retrieves an element from the buffer.  If the
// buffer is empty, the calling thread is blocked.  It will return false 
// without removing an element if it notices the isDone flag is set.
//
// put(T )  : bool
// A monitor function which puts an element in the buffer.  If the buffer
// is full, the calling thread is blocked until slots are available.  It
// will return false without adding an element if it notices the isDone
// flag is set.
//
// tearDown()
// A non-monitor function used to asynchronously tear down the bounded buffer.
// It sets the isDone flag, and wakes up all blocked threads with the
// pthread_broadcast call so that they can notice the flag and exit.
// It does not deallocate system resources or memory.
//
// empty() : bool
// Non-monitor routine; returns true/false depending on state of queue.

// Needed for threading support
#include <pthread.h>

#ifndef BOUNDEDBUFFER
#define BOUNDEDBUFFER

template <class T>
class BoundedBuffer
{
	public:
		BoundedBuffer(unsigned short size);
		~BoundedBuffer();
		bool get(T&);
		bool put(T);
		void tearDown();
		inline bool empty(){ return isEmpty; }
	private:
		pthread_mutex_t mutex;
		pthread_cond_t notFull,notEmpty;
		bool isFull,isEmpty;

		unsigned short head,tail;
		const unsigned short bufferSize;
		T* circle;
		bool isDone;
};

template <class T>
BoundedBuffer<T>::BoundedBuffer(unsigned short size) : bufferSize(size)
{
	circle = new T[bufferSize];
	head=tail=0;
	isFull=false;isEmpty=true;

	isDone=false;

	pthread_mutex_init(&mutex,NULL);
	pthread_cond_init(&notFull,NULL);
	pthread_cond_init(&notEmpty,NULL);
}

template <class T>
BoundedBuffer<T>::~BoundedBuffer()
{
	delete[] circle;
	pthread_mutex_destroy(&mutex);
	pthread_cond_destroy(&notFull);
	pthread_cond_destroy(&notEmpty);
}

template <class T>
bool BoundedBuffer<T>::get(T& target)
{
	pthread_mutex_lock(&mutex);

	while(isEmpty){
		pthread_cond_wait(&notEmpty,&mutex);

		if(isDone){
			pthread_mutex_unlock(&mutex);
			return false;
		}
	}

	if(isDone){
		pthread_mutex_unlock(&mutex);
		return false;
	}

	target=circle[head++];
	if(head==bufferSize) head=0;
	if(head==tail) isEmpty=true;
	isFull=false;
	pthread_cond_signal(&notFull);

	pthread_mutex_unlock(&mutex);

	return true;
}

template <class T>
bool BoundedBuffer<T>::put(T newVal)
{
	pthread_mutex_lock(&mutex);

	while(isFull)
	{
		pthread_cond_wait(&notFull,&mutex);

		if(isDone){
			pthread_mutex_unlock(&mutex);
			return false;
		}
	}

	if(isDone){
		pthread_mutex_unlock(&mutex);
		return false;
	}

	circle[tail++]=newVal;
	if(tail==bufferSize) tail=0;
	if(head==tail) isFull=true;
	isEmpty=false;
	pthread_cond_signal(&notEmpty);

	pthread_mutex_unlock(&mutex);

	return true;
}

template <class T>
void BoundedBuffer<T>::tearDown()
{
	isDone=true;
	pthread_cond_broadcast(&notFull);
	pthread_cond_broadcast(&notEmpty);
}

#endif
