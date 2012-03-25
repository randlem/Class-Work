//   File:          parser.h
//   Author(s):     R. Lancaster
//
//   Contents:
//      Declaration of the Parser class.  The parser class defines
//      several other datatypes for its own use:  Fnptr, StructType,
//      Structure, ForwardRef.  It uses the STL stack class to
//      retain forward reference and structure nesting information.
//
//      This class is derived from Compiler and uses a bottom-up
//      approach for parsing the source program.

#ifndef PARSER_H
#define PARSER_H

#include <stack>
using std::stack;

#include "compiler.h"
#include "emitter.h"
#include "scanner.h"
#include "symbol.h"

// Define a type for the function pointers
class Parser;
typedef void (Parser::*FnPtr)();

class Parser : public Compiler {
public:
   Parser();                       // the default constructor
   void compile(const string &);   // arg = source file name
private:
   enum DeclLevel { GLOBAL, LOCAL };
   enum StructType { STelse, STif, STproc, STprogram, STwhile };

   void advance();                        // get next triple
   void compileDeclarations(DeclLevel);   // compile declaration list
   void compileProcedures();              // compile procedures & main program
   PSymbol findSymbol(const string &);    // get Symbol pointer from tables
   void loadConstant(Register, PLiteral); // load literal into register
   void loadOperand(Register);            // load current operand

   // Declare code generator functions
   void AR(); void BR(); void CA(); void CM(); void CN(); 
   void EB(); void IO(); void LD(); void NO(); void PR();
   void ST(); void SU(); void xx();

   // Local variables
   static FnPtr CONO[15][15];      // action table for compiling procedures
   Register index;                 // index register for next load/store
   long nextLocation;              // next local variable to allocate
   Emitter sips;                   // code emitter
   Scanner source;                 // lexical analysis
   enum {CONTINUE, FREEZE, EXIT} status; // for compiling procedures
   bool suppressAdvance;           // skip advance() next time
   PTable symbolTable[2];          // array of symbol table pointers

   // The following values are set by advance()
   PToken currentOp;               // the current operator
   PToken nextOp;                  // the next operator
   POperand operandPtr;            // the pointer to the operand
   long operandAddress;            // address of the operand
   PLiteral literalPtr;            // the pointer to the literal operand
   int literalValue;               // value of the current literal
   PSymbol symbolPtr;              // the pointer to the symbol operand
   string symbolName;              // name of current symbol
   SymType symbolType;             // type of current symbol

   // The structure stack contains addresses of transfer vectors used
   // within if statements and while loops.  It is also used to verify
   // that controls are nested properly.
   class Structure {
   public:
      Structure(StructType st) { stType = st; }
      StructType stType;   // type of structure
      long conditionLoc;   // address of code to evaluate condition
      long jumpLoc;        // address of forward jump transfer vector
   };
   typedef Structure *PStructure;
   stack<PStructure> structStack;

   // The forward reference stack keeps track of references to symbols
   // (procedures) whose address is not yet known.
   class ForwardRef {
   public:
      ForwardRef(PSymbol op, long loc)
         { reference = op; instrLocation = loc; }
      PSymbol reference;   // pointer to Symbol of forward reference
      long instrLocation;  // address of transfer vector
   };
   typedef ForwardRef *PForwardRef;
   stack<PForwardRef> referenceStack;
};
   
typedef Parser *PParser;

#endif

