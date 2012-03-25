#include <vector>
using std::vector;

#include <stack>
using std::stack;

#include "exception.h"

#ifndef REWINDLIST_H
#define REWINDLIST_H

enum EntryType {entryAdd, entryRemove};

template <class T>
class RewindEntry {
public:
	RewindEntry() { ; }
	RewindEntry(double t, T value, int oldPos, EntryType entryType) {
		this->t = t;
		this->value = value;
		this->oldPos = oldPos;
		this->entryType = entryType;
	}

	// a collection of useful operator overloads (most could be removed)
	bool operator == (double right) {
		return(t == right);
	}

	bool operator >= (double right) {
		return(t <= right);
	}

	bool operator <= (double right) {
		return(t <= right);
	}

	bool operator < (double right) {
		return(t < right);
	}

	bool operator > (double right) {
		return(t > right);
	}

	T value;
	double t;
	int oldPos;
	EntryType entryType;
private:
};

template <class T>
class RewindList {
public:
	RewindList() { ; }
	RewindList(T*,int);

	bool rewind(double);
	bool stepBack();

	int add(T,double);
	T remove(int,double);

	T& operator [] (int i) { return(list[i]); }

	int size() { list.size(); }

	typedef vector<T>::iterator iterator;

private:
	vector<T> list;
	stack< RewindEntry<T> > rewindStack;
};

template <class T>
RewindList<T>::RewindList(T* ary, int size) {
	// seed the list with initial values
	for(int i = 0; i < size; ++i) {
		list.push_back(ary[i]);
	}
}

template <class T>
int	RewindList<T>::add(T n, double t) {
	// push a new element on the back of the array
	list.push_back(n);

	// insert an event into our rewind stack so we can delete this during a rollback
	rewindStack.push(RewindEntry<T>(t,n,0,entryAdd));

	// return the index of the last element of the array
	return(list.size() - 1);
}

template <class T>
T RewindList<T>::remove(int pos, double t) {
	RewindEntry<T> old(t,list[pos],pos,entryRemove);
	iterator index = list.begin() + pos;

	// push an event onto the rewind stack so we can rewind this during a rollback
	rewindStack.push(old);

	if(list.size() <= 0 || index > list.end())
		return(NULL);

	// erase the item currently there
	list.erase(index);
	// insert the last item in the list there
	list.insert(index,list.back());
	// remove the last item from the list to remove the dupe
	list.pop_back();

	return(*index);
}

template <class T>
bool RewindList<T>::rewind(double t) {
	RewindEntry<T> top;
	iterator i;

	// loop until the rewind stack is empty or we've reached the first event
	// before time t
	while(!rewindStack.empty() && rewindStack.top().t > t) {
		stepBack();
	}

	stepBack();

	return(true);
}

template <class T>
bool RewindList<T>::stepBack() {
	RewindEntry<T> top;
	iterator i;
	double t;

	if(rewindStack.empty())
		return(false);

	// retrive the time on the top of the stack
	t = rewindStack.top().t;

	// rewind the event
	while(t == rewindStack.top().t) {
		// retrive the top of the stack
		top = rewindStack.top();

		switch(top.entryType) {
			case entryAdd: {
				// added an item to the end of the list so just pop it off
				list.pop_back();
			}break;
			case entryRemove: {
				// calculate the position to insert the old record
				i = list.begin() + top.oldPos;

				// push the item to the back of the list
				list.push_back(list[top.oldPos]);

				// erase the item
				list.erase(i);

				// restore the previous item
				list.insert(i,top.value);
			}break;
		}

		// pop the top of the rewind stack
		rewindStack.pop();

	}

	// return something hopeful
	return(true);
}

#endif
