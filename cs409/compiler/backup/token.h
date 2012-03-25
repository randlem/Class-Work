//   File:          token.h
//   Author(s):     R. Lancaster
//
//   Contents:
//      Declaration of the Token class. Operand inherits from Token.
//      Token inherits from TableEntry so that tables can be made for
//      classes of Token objects (e.g. literals, symbols, operators).

#ifndef TOKEN_H
#define TOKEN_H

#include <string>
using std::string;

#include "table.h"  // Token inherits from TableEntry

//  This is the list of token categories for the language.
enum TokenType {
             TKarith,       // +, -, *, /
             TKleft,        // ( and [
             TKright,       // ) and ]
             TKsemicolon,   // ;
             TKrelOp,       // <, <=, =
             TKcomma,       // ,
             TKassign,      // ->
             TKbreak,       // break
             TKcall,        // call
             TKelse,        // else
             TKend,         // endif, endproc, endprogram, endwhile
             TKifWhile,     // if and while
             TKint,         // int
             TKprocProgram, // proc and program
             TKreadWrite,   // read and write
             TKendFile,     // end-of-file
             TKliteral,     // numeric literal
             TKsymbol       // symbol
             };

class Token : public TableEntry {
public:
   Token(TokenType tokenType, const string &lexeme)
      : tokenType(tokenType), lexeme(lexeme) { }
   virtual ~Token() { } // virtual destructor for base class
   string getKey() const { return lexeme; }
   TokenType getTokenType() const { return tokenType; }
private:
   TokenType tokenType;
   string lexeme;
};

typedef Token *PToken;
   
#endif

