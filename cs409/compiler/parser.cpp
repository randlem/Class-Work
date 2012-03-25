#include <fstream>
using std::ofstream;
using std::endl;

#include <iomanip>
using std::hex;
using std::ios;
using std::setw;
using std::setiosflags;
using std::setfill;

#include "parser.h"

void Parser::AR() {
	if(debugFlag) debug << "AR()" << endl;

	// load the new operand into REGt1 as the right-side of the op
	loadOperand(REGt1);

	// process the currentOp depending on it's key
	if(currentOp->getKey() == "+") {        // add
		sips.emit(OPadd,REGt0,REGt1,REGt0);
	} else if(currentOp->getKey() == "-") { // subtract
		sips.emit(OPsub,REGt0,REGt1,REGt0);
	} else if(currentOp->getKey() == "*") { // multiply
		sips.emit(OPmult,REGt0,REGt1,REGzero);
		sips.emit(OPmflo,REGzero,REGzero,REGt0);
	} else if(currentOp->getKey() == "/") { // divide
		sips.emit(OPdiv,REGt0,REGt1,REGzero);
		sips.emit(OPmflo,REGzero,REGzero,REGt0);
	}

	if(debugFlag) debug << endl;
}

void Parser::BR() {
	PStructure structure = NULL;

	if(debugFlag) debug << "BR()" << endl;

	// get the pointer to then pop the top structure
	structure = structStack.top();
	//structStack.pop();

	// a break is esentally a jump-on-register to the end of the structure...
	// however, this isn't going to work for en if-else syntax...as the break will
	// jump to the else section since i'm using the jumpLoc to store it's location
	// however, this should allow for break to act like return() if used in a
	// STproc or STprogram
	if(structure->stType != STproc) {
		sips.emit(OPlw,index,REGt3,structure->jumpLoc);
		sips.emit(OPjr,REGt3,REGzero,REGzero);
	} else {
		sips.emit(OPjr,REGra,REGzero,REGzero);

		// delete the LOCAL symbol table
		delete symbolTable[LOCAL];
		symbolTable[LOCAL] = NULL;
	}

	if(debugFlag) debug << endl;
}

void Parser::CA() {
	if(debugFlag) debug << "CA()" << endl;

	sips.emit(OPlw,index,REGt3,operandAddress);
	sips.emit(OPjalr,REGt3,REGzero,REGra);

	if(debugFlag) debug << endl;
}

void Parser::CM() {
	PStructure structure = structStack.top();
	long jumpLoc = (structure->stType == STelse) ? structure->elseLoc : structure->jumpLoc;

	if(debugFlag) debug << "CM()" << endl;

	// load the second operand into REGt1
	loadOperand(REGt1);

	// set the conditionLoc address to the REGpc
	structStack.top()->conditionLoc = sips.getPC() - 0x8;

	// depending on the copare operand do some different things
	if(currentOp->getKey() == "=") { // REGt0 = REGt1
		sips.emit(OPbeq,REGt0,REGt1,0xc);
		sips.emit(OPlw,index,REGt3,structure->jumpLoc);
		sips.emit(OPjr,REGt3,REGzero,REGra);
	} else if(currentOp->getKey() == "<") { // REGt0 < REGt1
		sips.emit(OPsub,REGt0,REGt1,REGt0);
		sips.emit(OPbltz,REGt0,REGzero,0xc);
		sips.emit(OPlw,index,REGt3,structure->jumpLoc);
		sips.emit(OPjr,REGt3,REGzero,REGra);
	} else { // REGt0 <= REGt1
		sips.emit(OPsub,REGt0,REGt1,REGt0);
		sips.emit(OPbltz,REGt0,REGzero,0x12);
		sips.emit(OPbeq,REGt0,REGzero,0xc);
		sips.emit(OPlw,index,REGt3,structure->jumpLoc);
		sips.emit(OPjr,REGt3,REGzero,REGra);
	}

	if(debugFlag) debug << endl;
}

void Parser::CN() {
	PStructure structure = new Structure(STif);

	if(debugFlag) debug << "CN()" << endl;

	// allocate the jumpLoc forward transfer vector and increment nextLocation
	structure->jumpLoc = nextLocation;
	nextLocation += 4;

	// if this is a while declaration then set the structure type to STwhile
	if(currentOp->getKey() == "while")
		structure->stType = STwhile;

	// push the structure on the top of the stack
	structStack.push(structure);

	if(debugFlag) debug << endl;
}

void Parser::EB() {
	PStructure structure = NULL; // general structure pointer

	if(debugFlag) debug << "EB()" << endl;

	// get the top structure on the stack and then pop it off
	structure = structStack.top();
	structStack.pop();

	// depending on the type of the structure process accordingly
	switch(structure->stType) {
		case STprogram: {
			// make sure this is the correct ending
			if(nextOp->getKey() != "endprogram")
				setError("Invalid nesting of structures! Did you mean to use 'endprogram'?");

			// set the status to end so the program ends normally
			status = EXIT;

			// set the jumpLoc of this structure to REGpc
			sips.emit(structure->jumpLoc,sips.getPC());
			if(debugFlag) debug << structure->jumpLoc << " " << sips.getPC() << endl;
		} break;
		case STproc: {
			// make sure this is the correct ending
			if(nextOp->getKey() != "endproc")
				setError("Invalid nesting of structures! Did you mean to use 'endproc'?");

			// delete our local symbol table
			if(symbolTable[LOCAL] != NULL) {
				delete symbolTable[LOCAL];
				symbolTable[LOCAL] = NULL;
			}

			// jump to the address stored in $ra
			sips.emit(OPjr,REGra,REGzero,REGzero);

			// put the address of the PC in the dataspace pointed to by mainJump
			sips.emit(mainJump,sips.getPC());
		} break;
		case STwhile: {
			// make sure this is the correct ending
			if(nextOp->getKey() != "endwhile")
				setError("Invalid nesting of structures! Did you mean to use 'endwhile'?");

			// set the jumpLoc to the REGpc + 4
			sips.emit(structure->jumpLoc,sips.getPC() + 4);

			// unconditionally jump to conditionLoc
			sips.emit(OPj,structure->conditionLoc);
		} break;
		case STif: {
			// if the nextOp is else then we're going to make the structure of type
			// STelse and put it back on the stack
			if(nextOp->getKey() == "else") {
				// allocate the elseLoc and increment nextLocation
				structure->elseLoc = nextLocation;
				nextLocation += 4;

				// change the structure type and push it back on the stack
				structure->stType = STelse;
				structStack.push(structure);

				// jump to the end of the else block pointed to by elseLoc
				sips.emit(OPlw,index,REGt3,structure->elseLoc);
				sips.emit(OPjr,REGt3,REGzero,REGra);
			} else if(nextOp->getKey() != "endif")
				setError("Invalid nesting of structures! Did you mean to use 'endif'?");

			// set the jumpLoc to the current PC
			sips.emit(structure->jumpLoc,sips.getPC());
		} break;
		case STelse: {

			// set the elseLoc to the current PC
			sips.emit(structure->elseLoc,sips.getPC());
		} break;
		default: {
			setError("Couldn't identify structure type!");
		} break;
	}

	if(debugFlag) debug << endl;
}

void Parser::IO() {
	PSymbol symbol = NULL;  // general symbol pointer

	if(debugFlag) debug << "IO()" << endl;

	// get the newline symbol
	symbol = findSymbol("newline");

	// make sure we freeze the scan
	status = FREEZE;

	// do different actions for read/write
	if(currentOp->getKey() == "read") {
		// loop and read variables till we reach the semicolon
		while(currentOp->getTokenType() != TKsemicolon) {
			sips.emit(OPaddi,REGzero,REGv0,0x5);
			sips.emit(OPsyscall,REGzero,REGzero,REGzero);
			sips.emit(OPsw,index,REGv0,operandAddress);
			advance();
		}
	} else {
		// loop and write variables till we reach the semicolon
		while(currentOp->getTokenType() != TKsemicolon) {
			if(nextOp->getTokenType() == TKleft)
				SU();
			loadOperand(REGa0);
			sips.emit(OPaddi,REGzero,REGv0,0x1);
			sips.emit(OPsyscall,REGzero,REGzero,REGzero);
			sips.emit(OPlw,index,REGa0,symbol->getAddress());
			sips.emit(OPaddi,REGzero,REGv0,0x4);
			sips.emit(OPsyscall,REGzero,REGzero,REGzero);
			advance();
		}

		// print a newline at the end of this line

	}

	if(debugFlag) debug << endl;
}

void Parser::LD() {
	if(debugFlag) debug << "LD()" << endl;

	// load the current operand into REGt1
	loadOperand(REGt0);

	if(debugFlag) debug << endl;
}

void Parser::NO() {
	if(debugFlag) debug << "NO()" << endl;
	// do nothing...ever...
	if(debugFlag) debug << endl;
}

void Parser::PR() {
	if(debugFlag) debug << "PR()" << endl;

	PStructure structure = new Structure(STprogram); // new structure for the stack

	// set the address of symbolPtr to nextLocation
	symbolPtr->setAddress(nextLocation);
	nextLocation += 4;

	// store a the curren REGpc in the address of our new symbol
	sips.emit(symbolPtr->getAddress(),sips.getPC());

	// set the symbol type of the program symbol
	symbolPtr->setSymbolType(SYprogram);

	// do some debug output
	if(debugFlag) {
		debug << operandPtr->getKey() << " "
		      << hex << "0x" << setw(8)
			  << setfill('0') << operandPtr->getAddress()
			  << setfill(' ') << " " << symbolPtr->getSymbolTypeString()
			  << endl;
	}

	// see if this is a proc definition, if so then change some things
	if(currentOp->getKey() == "proc") {
		structure->stType = STproc;
		symbolPtr->setSymbolType(SYproc);
	}

	// allocate the jumpLoc to the nextLocation and increment nextLocation
	sips.emit(symbolPtr->getAddress(),sips.getPC());
	structure->jumpLoc = nextLocation;
	nextLocation += 4;

	// push a Structure object of type STprogram on the structure stack
	structStack.push(structure);

	// see if the next token is an int, if so then call compileDeclarations() with
	// the correct level
	if(source.peekToken()->getTokenType() == TKint) {
		if(structure->stType == STproc)
			compileDeclarations(LOCAL);
		else
			compileDeclarations(GLOBAL);
	}

	if(debugFlag) debug << endl;
}

void Parser::ST() {
	if(debugFlag) debug << "ST()" << endl;

	// store the value of REGt0 in the address pointed to in operandAddress
	sips.emit(OPsw,index,REGt0,operandAddress);

	if(debugFlag) debug << endl;
}

void Parser::SU() {
	PToken ary_pos = NULL; // pointer to hold the address of the array position


	if(debugFlag) debug << "SU()" << endl;

	// set the status to FREEZE
	status = FREEZE;

	// read the next token for the array position
	ary_pos = source.nextToken();

	// make sure this token is of the correct type for a array subscript
	if(ary_pos->getTokenType() != TKsymbol && ary_pos->getTokenType() != TKliteral)
		setError("Invalid array subscript! " + ary_pos->getKey());

	// load the array position into REGt2
	if(ary_pos->getTokenType() == TKliteral) {
		loadConstant(REGt2,(PLiteral)ary_pos);
	} else {
		// find the symbol in the symbol table
		PSymbol symbol = findSymbol(ary_pos->getKey());

		// emit the code to load the operand into memory
		sips.emit(OPlw,index,REGt2,symbol->getAddress());
	}

	// shift the array position left by 2 (multiply by 4)
	sips.emit(OPsll,REGzero,REGt2,REGt2,0x2);

	// add the contents of REGgp
	sips.emit(OPadd,REGgp,REGt2,REGt2);

	// set index to REGt2
	index = REGt2;

	// put the next token in nextOp
	source.nextToken();
	nextOp = source.nextToken();

	if(debugFlag) debug << endl;
}

void Parser::xx() {
	string error = "";

	if(debugFlag) debug << "xx()" << endl;

	error = "Syntax Error! Offending code: ";

	if(currentOp != NULL)
		error += currentOp->getKey() + " ";

	if(operandPtr != NULL)
		error += currentOp->getKey() + " ";

	if(nextOp != NULL)
		error += currentOp->getKey();

	// we've got a syntax error
	setError(error);

	if(debugFlag) debug << endl;
}

Parser::Parser() {
	// set the initial status to continue
	status = CONTINUE;

	// init some variables with initial values
	index = REGgp;
	nextLocation = 0;
	symbolTable[LOCAL] = NULL;
	symbolTable[GLOBAL] = NULL;

	// allocate the GLOBAL scope symbol table
	symbolTable[GLOBAL] = new Table();
}

Parser::~Parser() {
	// if the LOCAL scope symbol table is allocated so delete it
	if(symbolTable[LOCAL] != NULL) {
		delete symbolTable[LOCAL];
	}

	// if the GLOBAL scope symbol table is allocated so delete it
	if(symbolTable[GLOBAL] != NULL) {
		delete symbolTable[GLOBAL];
	}

	// if the debug flag is set, then close the file
	if(debugFlag) {
		debug.close();
	}
}

void Parser::compile(const string &file) {
	string objectFileExt = ".spo";  // define a string version of the object file extension
	string objectFile    = changeFileExt(file,objectFileExt); // use changeFileExt() to get the object filename
	PToken tok           = NULL;    // a pointer to a token object
	PSymbol newSymbol    = NULL;    // a pointer to a symbol object

	// open the debug file if the debug flag is set
	if(debugFlag) {
		debug.open("debug.txt");
	}

	// open the source file
	source.openSourceFile(file);

	// open the object file
	sips.openObjectFile(changeFileExt(file,string(".spo")));

	// put a literal of '\n' in the program
	newSymbol = new Symbol("newline");
	newSymbol->setSymbolType(SYvar);
	newSymbol->setAddress(nextLocation);
	nextLocation += 4;
	sips.emit(newSymbol->getAddress(),nextLocation+0x44000);
	symbolTable[GLOBAL]->insert(newSymbol);
	sips.emit(nextLocation,0x0A000000);
	nextLocation += 4;

	// prime the pump by peeking the first token in the source file
	currentOp = source.peekToken();
	nextOp = source.peekToken();

	// call advance() until we think we find the beginning of the program format of
	// a token of TKprocProgram and then a token of TKsemicolor, or the EOF which ever
	// come first
	do {
		advance();
	} while(!((currentOp->getTokenType() == TKprocProgram &&
	           nextOp->getTokenType() == TKsemicolon) ||
			   nextOp->getTokenType() == TKendFile));

	// check to see if any error conditions were met
	if(nextOp->getTokenType() == TKendFile) {
		if(debugFlag)
			debug << "Could not find well formed beginning of program!" << endl;
		setError("Could not find beginning of program!");
	} else if(symbolPtr == NULL) {
		if(debugFlag)
			debug << "Could not find well formed beginning of program!" << endl;
		setError("Missing program name!");
	}

	// call the correct code generator to compile the "program [symbol] ;" token string
	PR();

	// peek ahead and see if there is a global int declaration, if so call
	// compileDeclaration with GLOBAL
	//tok = source.peekToken();
	//if(tok->getTokenType() == TKint) {
	//	compileDeclarations(GLOBAL);
	//}

	// call compileProcedures to compile the body of the program
	compileProcedures();

	// output code to terminate the MINI program
	sips.emit(OPaddi,REGzero,REGv0,0xA);
	sips.emit(OPsyscall,REGzero,REGzero,REGzero);

	if(debugFlag) debug << nextLocation << endl;

	// close the object file
	sips.closeObjectFile();

}

void Parser::advance() {
	PToken nextTok = NULL; // pointer to a token object

	// set the currentOp to the nextOp
	currentOp = nextOp;

	// get the next token from the source file
	nextTok = source.nextToken();

	// clear out the various pointers
	operandPtr = NULL;
	symbolPtr = NULL;
	literalPtr = NULL;

	// see if the nextTok is a symbol or literal
	if(nextTok->getTokenType() == TKsymbol) {
		// set the symbolPtr and operandPtr to the correct addresses
		symbolPtr = (PSymbol)findSymbol(nextTok->getKey());
		if(symbolPtr == NULL)
			setError("Undeclared variable used in expression.");
		operandPtr = (POperand)symbolPtr;

		// extract some useful data from the objects
		operandAddress = operandPtr->getAddress();
		symbolName = symbolPtr->getKey();
		symbolType = symbolPtr->getSymbolType();

		// get the next token
		nextTok = source.nextToken();
	} else if(nextTok->getTokenType() == TKliteral) {
		// set the literalPtr and operandPtr to the correct addresses
		literalPtr = (PLiteral)nextTok;
		operandPtr = (POperand)literalPtr;

		// extract some useful data from the objects
		operandAddress = operandPtr->getAddress();
		literalValue = literalPtr->getValue();

		// get the next token
		nextTok = source.nextToken();
	}

	// set the nextOp to the next token
	nextOp = nextTok;

	// if the debugFlag is set then do some debug output
	if(debugFlag) {
		debug << setiosflags(ios::left);
		debug << setw(10) << currentOp->getKey().c_str() << " ";
		if(operandPtr) {
			debug << setw(10) << operandPtr->getKey().c_str();
			debug << " (";
			debug << setiosflags(ios::right) << setfill('0');
			debug << hex << "0x" << setw(8) << operandPtr->getAddress();
			debug << setfill(' ') << setiosflags(ios::left) <<") ";
		} else {
			debug << "                        ";
		}
		debug << setw(10) << nextOp->getKey().c_str() << endl;
	}
}

void Parser::compileDeclarations(DeclLevel lvl) {
	PToken nextToken  = NULL;   // pointer to the next (current) token
	PSymbol newSymbol = NULL;   // the new symbol to insert
	long arySize      = 0;      // the dimension of the array if we have one
	int i             = 0;      // generic index

	// allocate the correct symbolTable at this level if there isn't one present
	if(symbolTable[lvl] == NULL)
		symbolTable[lvl] = new Table();

	// set nextToken to the current token
	nextToken = source.nextToken();

	// put the next token from the source in nextToken
	nextToken = source.nextToken();

	// until we get a semicolon (;) in nextToken, we're going to loop and process
	while(nextToken->getTokenType() != TKsemicolon) {

		// make sure it's a symbol, otherwise it's an error
		if(nextToken->getTokenType() != TKsymbol) {
			setError("Invalid token in variable declaration! (" +  nextToken->getKey() + ")" );
		}

		// allocate a new symbol object
		newSymbol = new Symbol(nextToken->getKey());

		// set the newSymbol address to nextLocation
		newSymbol->setAddress(nextLocation);

		// increment the value of nextLocation
		nextLocation += 4;

		// put the next token in nextToken
		nextToken = source.nextToken();

		// if this token is of type left ( '(' or '[' ) we're going to assume that it is an array
		if(nextToken->getTokenType() == TKleft) {

			// set the type of newSymbol to SYarray
			newSymbol->setSymbolType(SYarray);

			// set nextToken to the next token
			nextToken = source.nextToken();

			// make sure it's a literal with the size of the array otherwise error
			if(nextToken->getTokenType() == TKliteral)
				arySize = string2long(nextToken->getKey());
			else
				setError("Invalid array size!");

			if(arySize < minArySize) {
				setError("Array " + newSymbol->getKey() + " declared too small (minArySize = 0x" + long2string(minArySize) + "!");
			} else if(arySize > maxArySize) {
				setError("Array " + newSymbol->getKey() + " declared too large (maxSize = 0x" + long2string(maxArySize) + ")!");
			}

			// set nextToken to the next token
			nextToken = source.nextToken();

			// make sure it's of type TKright ( ')' or ']' ) otherwise error
			if(nextToken->getTokenType() != TKright)
				setError("Array definition never terminated!");

			// increment nextLocation to the end of the array
			nextLocation += 4 * arySize;

			// set nextToken to the next token
			nextToken = source.nextToken();

			// if the nextToken was an equal sign, generate code to initialize the values of the array
			if(nextToken->getKey() == "=") {
				nextToken = source.nextToken();
				i = 0;
				while(nextToken->getTokenType() != TKright && nextToken->getTokenType() != TKsemicolon) {
					nextToken = source.nextToken();
					sips.emit(newSymbol->getAddress() + (4 * i),string2long(nextToken->getKey()));
					nextToken = source.nextToken();
					i++;
				}
				if(nextToken->getTokenType() == TKsemicolon)
					setError("Array initlization never closed!");

				nextToken = source.nextToken();
			}

		} else {
			// set the type of newSymbol to SYvar
			newSymbol->setSymbolType(SYvar);

			// if nextToken is an equal sign, generate code to initialize the value of newSymbol
			if(nextToken->getKey() == "=") {
				nextToken = source.nextToken();
				sips.emit(newSymbol->getAddress(),string2long(nextToken->getKey()));
				nextToken = source.nextToken();
			}
		}

		// insert the new symbol into the table specified by lvl
		if(!symbolTable[lvl]->isPresent(newSymbol->getKey())) {
			symbolTable[lvl]->insert(newSymbol);
		} else {
			setError("Multiple defined symbol " + newSymbol->getKey() + "!");
		}

		// if the debugFlag is true then output a debug message
		if(debugFlag) {
			debug << newSymbol->getKey() << " "
			      << hex << "0x" << setw(8)
				  << setfill('0') << newSymbol->getAddress()
				  << setfill(' ') << " " << newSymbol->getSymbolTypeString()
				  << endl;
		}

		// put the next token from the source nextToken if nextToken is not a semicolon
		if(nextToken->getTokenType() != TKsemicolon)
			nextToken = source.nextToken();
	}
}

void Parser::compileProcedures() {
	status = CONTINUE;

	// if the next token is of type TKprocProgram then save the address of the
	// transfer vector that will hold the address of the first instruction of
	// the main program
	if(source.peekToken()->getTokenType() == TKprocProgram) {
		mainJump = nextLocation;
		nextLocation += 4;
		sips.emit(OPlw,index,REGt3,mainJump);
		sips.emit(OPjr,REGt3,REGzero,REGzero);
	}

	// loop until the status is EXIT
	while(status != EXIT) {
		// if the status is not FREEZE then call advance
		if(status != FREEZE) {
			advance();
		} else {
			status = CONTINUE;
		}

	    // use the CONO table to call right code generator based on currentOP and nextOP
		(this->*CONO[(int)currentOp->getTokenType()][(int)nextOp->getTokenType()])();
	}
}

PSymbol Parser::findSymbol(const string &s) {
	PSymbol newSymbol = NULL;  // the pointer to the new symbol

	// if the LOCAL symbol table is defined, then check it first
	if(symbolTable[LOCAL] != NULL) {
		if(symbolTable[LOCAL]->isPresent(s)) {
			if(debugFlag) debug << "LOCAL: " << s << endl;
			return((PSymbol)symbolTable[LOCAL]->find(s));
		}
	}

	// see if the symbol is in the GLOBAL symbol table
	if(symbolTable[GLOBAL]->isPresent(s)) {
		if(debugFlag) debug << "GLOBAL: " << s << endl;
		return((PSymbol)symbolTable[GLOBAL]->find(s));
	}

	// the symbol was in neither symbol table
	// so allocate a new symbol object
	newSymbol = new Symbol(s);
	newSymbol->setAddress(nextLocation);
	nextLocation += 4;

	// insert the new symbol object into the GLOBAL symbol table
	symbolTable[GLOBAL]->insert(newSymbol);

	// return the address of the new symbol
	return((PSymbol)symbolTable[GLOBAL]->find(s));
	//return(NULL);
}

void Parser::loadConstant(Register reg, PLiteral literal) {

	// if the value of the literal is less then or equal to
	// the largest number that a I-type instruction can hold
	// load the literal with an I-type instruction
	if(literal->getValue() < 0x7FFF) {
		sips.emit(OPaddi,REGzero,reg,literal->getValue());
		return;
	}

	// see if the literal is allocated in memory, if it is not then put the
	// value of the literal into nextLocation and increment nextLocation
	if(!literal->allocated()) {
		sips.emit(nextLocation,literal->getValue());
		literal->setAddress(nextLocation);
		nextLocation += 4;
	}

	// generate the approiate load for the large literal
	sips.emit(OPlw,index,reg,literal->getAddress());
}

void Parser::loadOperand(Register reg) {

	// if the operand is of type literal, then load a literal, else load the value
	// of the symbol into reg
	if(operandPtr->getTokenType() == TKliteral) {
		loadConstant(reg,(PLiteral)operandPtr);
	} else {
		// generate code to load the symbol's value into the appropriate register
		sips.emit(OPlw,index,reg,operandAddress);
	}
	index = REGgp;
}

string Parser::changeFileExt(const string &file, const string &ext) {
	string newStr = "";  // the new filename

	// extract the string part up to the period
	newStr = file.substr(0,file.find(".",0));

	// concat the extension ext onto the end of the parsed filename
	newStr += ext;

	// return the new filename
	return(newStr);
}

long Parser::string2long(const string &s) {
	// call the std function strtol() using base 16
	return(strtol(s.c_str(),NULL,16));
}

FnPtr Parser::CONO[15][15] = {
//   + - * /         ( [         ) ]           ;      < = <=           ,
//        ->       break        call        else        end*    if while
//       int    proc program read write
//
// + or - or * or /
   {&Parser::AR,&Parser::SU,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::AR,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// ( or [
   {&Parser::xx,&Parser::SU,&Parser::xx,&Parser::xx,&Parser::LD,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// ) or ]
   {&Parser::LD,&Parser::SU,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::LD,&Parser::NO,&Parser::NO,&Parser::xx,&Parser::xx,&Parser::NO,
    &Parser::xx,&Parser::xx,&Parser::NO},
// ;
   {&Parser::LD,&Parser::SU,&Parser::xx,&Parser::NO,&Parser::xx,&Parser::xx,
    &Parser::LD,&Parser::NO,&Parser::NO,&Parser::EB,&Parser::EB,&Parser::NO,
    &Parser::xx,&Parser::NO,&Parser::NO},
// < or = or <=
   {&Parser::xx,&Parser::SU,&Parser::CM,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// ,
   {&Parser::xx,&Parser::SU,&Parser::xx,&Parser::IO,&Parser::xx,&Parser::IO,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// ->
   {&Parser::ST,&Parser::SU,&Parser::xx,&Parser::ST,&Parser::xx,&Parser::xx,
    &Parser::ST,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// break
   {&Parser::xx,&Parser::xx,&Parser::xx,&Parser::BR,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// call
   {&Parser::xx,&Parser::xx,&Parser::xx,&Parser::CA,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// else
   {&Parser::LD,&Parser::SU,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::LD,&Parser::NO,&Parser::NO,&Parser::xx,&Parser::xx,&Parser::NO,
    &Parser::xx,&Parser::xx,&Parser::NO},
// end...
   {&Parser::xx,&Parser::xx,&Parser::xx,&Parser::NO,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// if or while
   {&Parser::xx,&Parser::CN,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// int
   {&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// proc or program
   {&Parser::xx,&Parser::xx,&Parser::xx,&Parser::PR,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// read or write
   {&Parser::xx,&Parser::SU,&Parser::xx,&Parser::IO,&Parser::xx,&Parser::IO,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx}
//
//   + - * /         ( [         ) ]           ;      < = <=           ,
//        ->       break        call        else        end*    if while
//       int    proc program read write
};
