//   File:          compilerexception.h
//   Author(s):     R. Lancaster
//
//   Contents:
//      Declaration of the CompilerException class

#ifndef COMPILEREXCEPTION_H
#define COMPILEREXCEPTION_H

#include <string>
using std::string;

class CompilerException {
public:
   CompilerException(int lineNumber, string message)
      : lineNumber(lineNumber), message(message) { }
   int lineNumber;
   string message;
};

typedef CompilerException *PCompilerException;

#endif
