//   File:         operand.h
//   Author(s):    Ron Lancaster
//
//   Contents:
//      Declaration of the Operand class
//
//   Comments:
//      All member functions are defined inline.  Literal and
//      Symbol inherit from this class.
//
//      The constructor is protected because objects of type
//      Operand are never created directly.  This class is needed
//      because the parser needs to point to operand tokens, which
//      could be either symbols or literals.  Addresses are associated with
//      symbols and with large literals, so the address is stored here.

#ifndef OPERAND_H
#define OPERAND_H

#include <string>
using std::string;

#include "token.h"

class Operand : public Token {
public:
   bool allocated() const { return (address != -1); }
   long getAddress() const { return address; }
   void setAddress(long address) { this->address = address; }
protected:
   Operand(TokenType tt, const string &str) : Token(tt, str), 
      address(-1) { }
   long address;      // address for the operand
};
   
typedef Operand *POperand;

#endif

