//   File:         symbol.h
//   Author(s):    Ron Lancaster
//
//   Contents:
//      Declaration of the Symbol class
//
//   Comments:
//      This class inherits from Operand.
//

#ifndef SYMBOL_H
#define SYMBOL_H

#include <string>
using std::string;

#include "operand.h"

// This gives the data types for user-defined symbols.
enum SymType {SYvar, SYarray, SYproc, SYforwardProc, SYprogram, SYunknown};

class Symbol : public Operand {
public:
   Symbol(const string &newName) : Operand(TKsymbol, newName),
        thisSymbolType(SYunknown) { }
   SymType getSymbolType() const { return thisSymbolType; }
   void setSymbolType(SymType newSymType) { thisSymbolType = newSymType; }
private:
   SymType thisSymbolType; // the type of this symbol
};
   
typedef Symbol *PSymbol;

#endif

