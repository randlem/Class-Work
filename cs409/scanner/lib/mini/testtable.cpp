// tab.cpp:  Example showing use of Table class

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
   Table tabTable;
   tabPtr = new Tab("abc", 111);
   tabTable.insert(tabPtr);
   tabTable.insert(new Tab("abc2", 222));
   tabTable.insert(new Tab("abc3", 333));
   tabTable.insert(new Tab("abc4", 444));
   tabTable.insert(new Tab("abc5", 555));

   // Try to find object with key "abc"
   cout << "abc ";
   if (tabTable.isPresent("abc"))
      cout << "found" << endl;
   else
      cout << "not found" << endl;

   tabPtr = PTab(tabTable.find("abc"));
   if (tabPtr == NULL)
      cout << "object not found from find function" << endl;
   else
      cout << tabPtr->getValue() << endl;

   return 0;
}

