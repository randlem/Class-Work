//   File:          scan.cpp
//   Author(s):     R. Lancaster
//
//   Contents:
//      This file contains the main program used to test the
//      lexical analysis component of the compiler.

#include <exception>
using std::exception;

#include <iostream>
using std::cerr;
using std::cin;
using std::cout;
using std::endl;

#include <iomanip>
using std::setw;
using std::setiosflags;
using std::ios;

#include <fstream>
using std::ofstream;

#include <string>
using std::string;

#include "compilerexception.h"
#include "operand.h"
#include "scanner.h"
#include "token.h"

int main(int argc, char *argv[])
{
   bool more = true;   // controls loop
   ofstream outfile;   // output file scan.txt
   string fileName;
   Scanner source;
   PToken nextToken;

   /*static string tokenName[] =
        { "TKaddOp", "TKmultOp", "TKleft", "TKright", "TKlBrak", "TKrBrak",
        "TKsemicolon", "TKrelOp", "TKcomma", "TKassign", "TKcall",
        "TKelse", "TKend", "TKifWhile", "TKproc", "TKthenDo", "TKwrite",
        "TKendFile", "TKliteral", "TKsymbol"};*/

	static string tokenName[] =
		{    "TKarith",       // +, -, *, /
             "TKleft",        // ( and [
             "TKright",       // ) and ]
             "TKsemicolon",   // ;
             "TKrelOp",       // <, <=, =
             "TKcomma",       // ,
             "TKassign",      // ->
             "TKbreak",       // break
             "TKcall",        // call
             "TKelse",        // else
             "TKend",         // endif, endproc, endprogram, endwhile
             "TKifWhile",     // if and while
             "TKint",         // int
             "TKprocProgram", // proc and program
             "TKreadWrite",   // read and write
             "TKendFile",     // end-of-file
             "TKliteral",     // numeric literal
             "TKsymbol"       // symbol
		};

   // Get the name of the source file
   if (argc < 2) {
      cout << "Source program: ";
      cin >> fileName;
      }
   else
      fileName = argv[1];

   // Do nothing if no filename provided
   if (fileName == "")
      return 1;

   try {
      // Open source file
      source.openSourceFile(fileName);

      // Open output file
      outfile.open("scan.txt");
      if (!outfile) {
         cerr << "Unable to open scan.txt" << endl;
         return 1;
      }

      outfile << "Tokens in file " << fileName << endl << endl;
      do {
         nextToken = source.nextToken();
         outfile << setiosflags(ios::left) << setw(15)
                 << tokenName[nextToken->getTokenType()].c_str();

         switch (nextToken->getTokenType()) {
         case TKsymbol:
            outfile << nextToken->getKey()
                    << " (" << long(nextToken) << ')' << endl;
            break;
         case TKliteral:
            outfile << PLiteral(nextToken)->getKey()
                    << " (" << long(nextToken) << ')' << endl;
            break;
         default:
            more = nextToken->getTokenType() != TKendFile;
            outfile << nextToken->getKey() << endl;
         }
      }
      while (more);
   }
   catch (CompilerException exc) {
      cerr << "Error on line " << exc.lineNumber << ": "
           << exc.message << endl;
      if (outfile.is_open()) outfile.close();
      return 1;
   }
   catch (exception exc) {
      cerr << "System error: " << exc.what() << endl;
      if (outfile.is_open()) outfile.close();
      return 1;
   }

   outfile.close();
   return 0;
}

