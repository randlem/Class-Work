//   File:          table.h
//   Author(s):     R. Lancaster
//
//   Contents:
//      Declaration of a Table class for hash tables.
//
//   Comments:
//      This file defines two classes:  TableEntry and Table.  All
//      records to be added to the table must be classes derived from
//      TableEntry.  This will require the record to define a function
//      called "getKey" that returns the key of the record (a string).
//
//      The Table class implements a hash table containing records to
//      pointers.  This implementation doesn't allow a record to be deleted
//      from the table.

#ifndef TABLE_H
#define TABLE_H

#include <string>
using std::string;

class TableEntry {
public:
   virtual ~TableEntry() { }           // virtual destructor
   virtual string getKey() const = 0;  // pure virtual function
};
typedef TableEntry *PTableEntry;

class Table {
public:
   Table(int size=511);
   ~Table();
   void insert(PTableEntry);
   bool isPresent(const string &) const;
   PTableEntry find(const string &) const;
   short getCount( ) const { return count; }
private:
   // disallow copy constructor and assignment operator
   // for tables
   Table(const Table &);
   const Table &operator=(const Table &);

   // Private data
   PTableEntry *data;    // array of pointers
   const short capacity; // size of table
   short count;          // number of items in list
};
typedef Table *PTable;

#endif

