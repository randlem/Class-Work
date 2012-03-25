//   File:          scanner.h
//   Author(s):     R. Lancaster
//
//   Contents:
//      Declaration of the Scanner class

#ifndef SCANNER_H
#define SCANNER_H

#include <fstream>
using std::ifstream;

#include <iostream>
using std::endl;

#include "compiler.h"
#include "literal.h"
#include "table.h"
#include "token.h"

class Scanner : public Compiler {
public:
	Scanner();                      // default constructor
	~Scanner();                     // closes source file if open
	PToken nextToken() {
		if (peekFlag) {
		   peekFlag = false;
		   return saveToken;
		}
		//saveToken = NULL;
		saveToken = getToken();
		//if(debugFlag) debug << saveToken->getKey() << endl;
		return (saveToken);
	}
	PToken peekToken() {
		if (peekFlag)
			return(saveToken);
		peekFlag = true;
		saveToken = getToken();
		return (saveToken);
	}
	void openSourceFile(const string &); // arg = filename

private:
	static string char2string(char c) { return string(1, c); } // convert char to string
	PLiteral getLiteral(long);
	PToken getToken();
	PLiteral readLiteral(char);
	PToken readSymbol(char);

	bool peekFlag;                  // did we peek a token?
	ifstream sourceFile;            // the source program
	PToken saveToken;               // the previous token
	Table literalTable;             // table of all literals
	Table tokenTable;               // this table has operators, reserved
									// words, and symbols
};

typedef Scanner *PScanner;

#endif

