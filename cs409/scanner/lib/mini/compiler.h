//   File:          compiler.h
//   Author(s):     R. Lancaster
//
//   Contents:
//      Declaration of the Compiler class

#ifndef COMPILER_H
#define COMPILER_H

#include <fstream>
using std::ofstream;

#include <string>
using std::string;

//    Define a class to contain static data and functions needed throughout
// the compiler.  Classes Scanner, Parser, and Generator are derived
// from this class.
//    The only public data is debugFlag.  Static data ensures that all
// derived objects contain the same data values.  Two static functions
// are public (long2string, setError).
//    The copy constructor and assignment operator are disallowed
// (i.e. made private) because derived classes contain pointers.

class Compiler {
public:
   virtual ~Compiler() { }            // virtual destructor for base class
   static bool debugFlag;             // true if debug output wanted
   static string long2string(long);   // convert a long value to hex string
   static void setError(const string &); // report error, throw exception
protected:
   Compiler() { }                     // the default constructor

   static ofstream debug;             // debug output file
   static short lineNumber;           // source file line number
private:
   Compiler(const Compiler &);        // disallow copy constructor
   Compiler operator=(const Compiler &);  // disallow assignment
};

typedef Compiler *PCompiler;

#endif

