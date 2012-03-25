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

	bool rollback(double);

	bool add(T,double);
	bool remove(int,double);

	T& operator [] (int i) { return(list[i]); }

	int size() { list.size(); }

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
bool RewindList<T>::add(T n, double t) {
	// push a new element on the back of the array
	list.push_back(n);

	rewindStack.push(RewindEntry<T>(t,n,0,entryAdd));
}

template <class T>
bool RewindList<T>::remove(int pos, double t) {
	RewindEntry<T> old(t, list[pos], pos,entryRemove);
	vector<T>::iterator index = list.begin() + pos;

	rewindStack.push(old);
	list.erase(index);
	list.insert(index,list.back());
	list.pop_back();
}

template <class T>
bool RewindList<T>::rollback(double t) {
	RewindEntry<T> top;
	vector<T>::iterator i;

	while(!rewindStack.empty() && rewindStack.top().t >= t) {
		top = rewindStack.top();

		switch(top.entryType) {
			case entryAdd: {
				list.pop_back();
			}break;
			case entryRemove: {
				i = list.begin() + top.oldPos;
				list.push_back(list[top.oldPos]);
				list.erase(i);
				list.insert(i,top.value);
			}break;
		}

		rewindStack.pop();
	}

	return(true);
}

#endif
