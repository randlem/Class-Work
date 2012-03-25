// tab.cpp:  Example showing use of Table class

#include <stdio.h>

#include <iostream>
using std::cout;
using std::endl;

#include "table.h"

class Tab : public TableEntry {
public:
   Tab(const string &st, short sh) { key = st; tableValue = sh; }
   string getKey() const { return key; }
   short getValue() const { return tableValue; }

private:
   string key;
   short tableValue;
};
typedef Tab *PTab;

int main()
{
   PTab  tabPtr;
   Table tabTable(2044);
   tabPtr = new Tab("*", 111);
   tabTable.insert(tabPtr);
   tabTable.insert(new Tab("abc2", 222));
   tabTable.insert(new Tab("abc3", 333));
   tabTable.insert(new Tab("abc4", 444));
   tabTable.insert(new Tab("abc5", 555));

	for(int i=0; i < 1000; i++) {
		char* cp; string s;
		sprintf(cp,"asdf%i",i);
		s = cp;
		tabTable.insert(new Tab(s,i));
	}

	for(int i=0; i < 1000; i++) {
		char cp[80]; string s; PTableEntry e;
		sprintf(cp,"asdf%i",i);
		s = cp;
		if(tabTable.isPresent(s))
			cout << s << " found" << endl;
		else
			cout << s << " is not found" << endl;
		if((e = tabTable.find(s)) == NULL) {
			cout << s << " find failed" << endl;
		} else {
			cout << s << " " << e->getKey() << endl;
		}
	}

   // Try to find object with key "abc"
   cout << "* ";
   if (tabTable.isPresent("*"))
      cout << "found" << endl;
   else
      cout << "not found" << endl;

   tabPtr = PTab(tabTable.find("sdf"));
   if (tabPtr == NULL)
      cout << "object not found from find function" << endl;
   else
      cout << tabPtr->getValue() << endl;

	/*tabTable.printTable();*/
   cout << "Collisions: " << tabTable.getCollision() << endl;

   return 0;
}
