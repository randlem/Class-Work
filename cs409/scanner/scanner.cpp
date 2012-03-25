#include <string>
using std::string;

#include <fstream>
using std::ifstream;

#include "compiler.h"
#include "literal.h"
#include "table.h"
#include "token.h"
#include "scanner.h"

Scanner::Scanner() : peekFlag(false), saveToken(NULL) {
	// arithmetic operators
	tokenTable.insert(new Token(TKarith,"+"));
	tokenTable.insert(new Token(TKarith,"-"));
	tokenTable.insert(new Token(TKarith,"*"));
	tokenTable.insert(new Token(TKarith,"/"));

	// left brackets/braces
	tokenTable.insert(new Token(TKleft,"["));
	tokenTable.insert(new Token(TKleft,"("));

	// right brackets/braces
	tokenTable.insert(new Token(TKright,"]"));
	tokenTable.insert(new Token(TKright,")"));

	// semicolon
	tokenTable.insert(new Token(TKsemicolon,";"));

	// relational operators
	tokenTable.insert(new Token(TKrelOp,"<"));
	tokenTable.insert(new Token(TKrelOp,"<="));
	tokenTable.insert(new Token(TKrelOp,"="));

	// comma
	tokenTable.insert(new Token(TKcomma,","));

	// assignment operator
	tokenTable.insert(new Token(TKassign,"->"));

	// break operator
	tokenTable.insert(new Token(TKbreak,"break"));

	// call operator
	tokenTable.insert(new Token(TKcall,"call"));

	// else
	tokenTable.insert(new Token(TKelse,"else"));

	// end statements
	tokenTable.insert(new Token(TKend,"endif"));
	tokenTable.insert(new Token(TKend,"endproc"));
	tokenTable.insert(new Token(TKend,"endprogram"));
	tokenTable.insert(new Token(TKend,"endwhile"));

	// if and while statements
	tokenTable.insert(new Token(TKifWhile,"if"));
	tokenTable.insert(new Token(TKifWhile,"while"));

	// integer declarations
	tokenTable.insert(new Token(TKint,"int"));

	// proc and program statemetns
	tokenTable.insert(new Token(TKprocProgram,"proc"));
	tokenTable.insert(new Token(TKprocProgram,"program"));

	// read and write statements
	tokenTable.insert(new Token(TKreadWrite,"read"));
	tokenTable.insert(new Token(TKreadWrite,"write"));

	// end of file argument
	tokenTable.insert(new Token(TKendFile,"*EOF*"));

}

Scanner::~Scanner() {
	if(!sourceFile.is_open()) {
		sourceFile.close();
	}
}

void Scanner::openSourceFile(const string &s) {
	sourceFile.open(s.c_str());
	if(!sourceFile) {
		setError("Couldn't open source file " + s + "!");
	}
}

PLiteral Scanner::getLiteral(long l) {
	PLiteral literal;
	string s = long2string(l);

	if(!(literal = (PLiteral)literalTable.find(s))) {
		literalTable.insert(new Literal(l));
		return((PLiteral)literalTable.find(s));
	}
	return(literal);
}

PToken Scanner::getToken() {
	char c;

	sourceFile.get(c);

	if(sourceFile.eof()) {
		return((PToken)tokenTable.find("*EOF*"));
	}

	while(c <= ' ') {
		if(c == '\n') {
			lineNumber++;
		}
		sourceFile.get(c);

		if(sourceFile.eof()) {
			return((PToken)tokenTable.find("*EOF*"));
		}
	}

	// found a symbol
	if((((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'z'))) && (c != '[')) {
		return((PToken)readSymbol(c));
	}

	// found a non-negative literal
	if((c >= '0') && (c <= '9')) {
		return((PToken)readLiteral(c));
	}

	// found a character literal
	if(c == '\'') {
		sourceFile.get(c);
		PLiteral literal = getLiteral((long)c);
		sourceFile.get(c);
		return((PToken)literal);
	}

	// found either a comment of a divide token
	if(c == '/') {
		sourceFile.get(c);
		if(c == '*') {
			sourceFile.get(c);
			while(c != '*') {
				if(c == '\n') lineNumber++;
				sourceFile.get(c);
				if(sourceFile.eof()) {
					return((PToken)tokenTable.find("*EOF*"));
				}
			}
			sourceFile.get(c);
			return((PToken)getToken());
		} else {
			sourceFile.putback(c);
			return((PToken)tokenTable.find("/"));
		}
	}

	// either found the <= token or a < sign
	if(c == '<') {
		char c1;
		sourceFile.get(c1);
		if(c1 == '=') {
			return((PToken)tokenTable.find("<="));
		} else {
			sourceFile.putback(c1);
			return((PToken)tokenTable.find("<"));
		}
	}

	// either found a -> token, a negative literal, or a - sign
	if(c == '-') {
		char c1;
		sourceFile.get(c1);
		if(c1 == '>') {
			return((PToken)tokenTable.find("->"));
		} else if((c1 >= '0') && (c1 <= '1')) {
			sourceFile.putback(c1);
			return((PToken)readLiteral(c));
		} else {
			sourceFile.putback(c1);
			return((PToken)tokenTable.find("-"));
		}
	}

	if(!tokenTable.isPresent(char2string(c))) {
		setError("Invalid Symbol!");
	}

	return((PToken)tokenTable.find(char2string(c)));
}

PLiteral Scanner::readLiteral(char c) {
	long long literal=0;
	int base=0,neg=false;

	if(c == '-') {
		neg = true;
		base = 10;
	} else {
		if(c == '0') {
			sourceFile.get(c);
			if(c == 'x' || c == 'X') {
				base = 16;
			} else if(c >= '0' && c <='9') {
				sourceFile.putback(c);
				base = 8;
			} else {
				sourceFile.putback(c);
				return(getLiteral(0));
			}
		} else {
			sourceFile.putback(c);
			base = 10;
		}
	}

	if(base==10 || base==8) {
		while(true) {
			sourceFile.get(c);
			if(c == ' ' || c == '\n'  || c == '\t' || c == ',' || c == ';' || c == ']' || c == ')' || c == '-') {
				sourceFile.putback(c);
				break;
			}
			if(!(c >= '0' && c <= '9')) {
				setError("Invalid Character in Literal object!");
			}
			if(base == 8 && c >= '8') {
				setError("Invalid Character in Literal object!");
			}
			c-='0';
			literal *= base;
			literal += c;
		}
	} else {
		while(true) {
			sourceFile.get(c);
			if(c == ' ' || c == '\n'  || c == '\t' || c == ',' || c == ';') {
				sourceFile.putback(c);
				break;
			}
			if(c >= '0' && c <= '9') {
				c -= '0';
			} else if(c >= 'a' && c <= 'f') {
				c -= 'a';
				c += 10;
			} else if(c >= 'A' && c <= 'F') {
				c -= 'A';
				c += 10;
			} else {
				setError("Invalid character in Literal object!");
			}
			literal *= base;
			literal += c;
		}
	}

	if(literal > 4294967295) {
		setError("Literal Overflow!");
	}

	if(neg) {
		literal *= -1;
	}

	return(getLiteral((long)literal));
}

PToken Scanner::readSymbol(char c) {
	string symbol = char2string(c);
	PToken token = NULL;

	while(true) {
		sourceFile.get(c);
		if(!(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9')) {
			sourceFile.putback(c);
			break;
		}
		symbol += char2string(c);
	}

	if(token = (PToken)tokenTable.find(symbol)) {
		return(token);
	}

	tokenTable.insert(new Token(TKsymbol,symbol));

	return((PToken)tokenTable.find(symbol));
}
