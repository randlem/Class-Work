//   File:         literal.h
//   Author(s):    Mark Randles
//
//   Contents:
//      Declaration of the Literal class
//
//   Comments:
//      All member functions are defined inline.
//
//      This class is needed to represent literal tokens gathered
//      by the scanner class during compile time.  Derived from type
//      Operand, it stores a absolute value of the literal and uses
//      it as it's key, by calling Compiler::long2str() on the long
//      value passed into it's constructor.

#ifndef LITERAL_H
#define LITERAL_H

#include "compiler.h"
#include "operand.h"

class Literal : public Operand {
	public:
		Literal(const long l) :
			Operand(TKliteral,Compiler::long2string(l)),
			value(l) { }
		long getValue() const { return value; }

	private:
		long value;
};
typedef Literal *PLiteral;

#endif
