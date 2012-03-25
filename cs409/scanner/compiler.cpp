#include <fstream.h>
using std::ofstream;
using std::endl;

#include <string.h>
using std::string;

#include "compiler.h"
#include "compilerexception.h"

bool Compiler::debugFlag=false;
ofstream Compiler::debug;
short Compiler::lineNumber = 1;

string Compiler::long2string(long l) {
	string s;

	for(int i=7; i >= 0; i--) {
		char c = 0x0000000F & (l >> (i * 4));
		s += ((c < 0xA) ? '0' + c : 'A' + c - 0xA);
	}

	return(s);
}

void Compiler::setError(const string &err) {
	throw CompilerException(lineNumber,err);
}
