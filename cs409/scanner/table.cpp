#include <iostream>
using std::cout;
using std::endl;

#include <memory>

#include <string>
using std::string;

#include "table.h"

Table::Table(int size) : capacity(size) {
	// initilize count to zero since we have no
	// table entries
	count = 0;

	// allocated the array pointed to by data and
	// initilize the pointers to NULL through a memset()
	data = new PTableEntry[capacity];
	memset(data,NULL,sizeof(PTableEntry) * capacity);
}

Table::~Table() {
	int i;   // counter

	// deallocate any entry left in the table
	for(i=0; i < capacity; ++i) {
		delete data[i];
		data[i] = NULL;
	}

	// delete our array
	delete [] data;
	data = NULL;

	// set final value for count
	count = 0;
}

void Table::insert(PTableEntry new_e) {
	int hash_key = hashKey(new_e->getKey()) % capacity;
	PTableEntry e = data[hash_key];

	// insert the new entry
	if(e == NULL) {
		// nothing is at the hash bucket so just
		// insert the value
		data[hash_key] = new_e;
	} else {

		// create another entry pointer and set it equal to the
		// the first link
		PTableEntry next_e = e->getLink();
		while(next_e != NULL) {
			e = next_e;
			next_e = e->getLink();
		}
		// we've found the end of the list so insert our new value
		e->setLink(new_e);
		new_e->setLink(NULL);
	}

}

bool Table::isPresent(const string &key) const {
	PTableEntry e;

	// if there is something at e, make sure it's the
	// right value
	if(e = data[hashKey(key) % capacity]) {
		// since the first link wasn't our key, loop till we
		// run out of links or find it
		while(e->getKey() != key) {
			e = e->getLink();

			// ran out of links return false, key isn't in here
			if(e == NULL) {
				return(false);
			}
		}

		// matched the keys so return true
		return(true);
	}

	// nothing at that hash spot so return false
	return(false);
}

PTableEntry Table::find(const string &key) const {
	PTableEntry e = data[hashKey(key) % capacity];

	// loop till we find our node or run out of links
	if(e = data[hashKey(key) % capacity]) {
		while(e->getKey() != key) {
			e = e->getLink();
			if(e == NULL) {
				break;
			}
		}
	}

	// if we didn't find key or ran out of links e is set to NULL
	// otherwise we are going to return the address of the table
	// entry that matches key
	return(e);
}

int Table::hashKey(const string &s) const {
	char *p;
	unsigned int h, g;

	h = 0;
	for(p=(char *)s.c_str(); *p!='\0'; p++){
		h = (h<<4) + *p;
		if(g = h&0xF0000000){
			h ^= g>>24;
			h ^= g;
		}
	}

	return h;
}

/* alternate hash function */
/*int Table::hashKey(const string &s) const {
	char* p = (char*)s.c_str();
	unsigned int h;
	int i, c, d, n;

	h = s.length();
	while(*p) {
		d = *p++;
		c = d;
		c ^= c<<6;
		h += (c<<11) ^ (c>>1);
		h ^= (d<<14) + (d<<7) + (d<<4) + d;
	}

	return(h);
}*/
