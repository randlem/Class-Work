//   File:          mini.cpp
//   Author(s):     R. Lancaster
//
//   Contents:
//      MINI Compiler Main Program

#include <iostream>
using std::cerr;
using std::cin;
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <exception>
using std::exception;

#include "compilerexception.h"
#include "parser.h"

int main(int argc, char *argv[])
{
   string response;
   string sourceFileName;
   Parser mini;

   // Get the name of the file containing the program to compile
   if (argc < 2) {
      // No filename was provided at startup
      cout << "File containing program to compile: ";
      cin >> sourceFileName;
      cout << "Debug output (y/n)? ";
      cin >> response;
      mini.debugFlag = ((response=="y") || (response=="Y"));
   }
   else {
      // Obtain filename from argument
      sourceFileName = argv[1];
      if (sourceFileName == "-d") {
         mini.debugFlag = true;
         if (argc > 2) sourceFileName = argv[2];
      }
   }

   // Do nothing if no filename was provided
   if (sourceFileName == "")
      return 1;

   // Compile the specified program
   try {
      mini.compile(sourceFileName);
      cout << "Successful compilation!" << endl;
   }
   catch (CompilerException exc) {
      cerr << "Error on line " << exc.lineNumber << ": "
           << exc.message << endl;
      return 1;
   }
   catch (exception exc) {
      cerr << "System error: " << exc.what() << endl;
      return 1;
   }
   
   return 0;
}

