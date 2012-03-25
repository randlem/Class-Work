//   File:         emitter.h
//   Author(s):    Ron Lancaster
//
//   Contents:
//      Declaration of the Emitter class for the SIPS Virtual
//      Machine.

#ifndef EMITTER_H
#define EMITTER_H

#include <fstream>
using std::ofstream;

#include "compiler.h"

// Define the register codes 
enum Register { REGv0= 2, REGv1= 3, REGa0= 4, REGa1= 5, REGa2= 6,
                REGa3= 7, REGt0= 8, REGt1= 9, REGt2=10, REGt3=11,
                REGs0=16, REGs1=17, REGs2=18, REGs3=19, REGgp=28,
                REGsp=29, REGfp=30, REGra=31, REGzero= 0 };

// Define the opcodes by type (I, J, R)
enum IOpcode {
   OPaddi    = 0x08,  // Add immediate
   OPaddiu   = 0x09,  // Add immediate unsigned
   OPandi    = 0x0c,  // Bitwise AND immediate
   OPbeq     = 0x04,  // Branch on equal
   OPbgtz    = 0x07,  // Branch on > zero
   OPbltz    = 0x01,  // Branch on < zero
   OPbne     = 0x05,  // Branch on not equal
   OPlh      = 0x21,  // Load halfword
   OPlw      = 0x23,  // Load word
   OPori     = 0x0d,  // Bitwise OR immediate
   OPsh      = 0x29,  // Store halfword
   OPslti    = 0x0a,  // Set less than immediate
   OPsw      = 0x2b,  // Store word
   OPxori    = 0x0e,  // Bitwise XOR immediate
};

enum JOpcode {
   OPj       = 0x02,  // Unconditional jump
   OPjal     = 0x03,  // Jump and link
};

enum ROpcode {
   OPadd     = 0x20,  // Add
   OPaddu    = 0x21,  // Add unsigned
   OPand     = 0x24,  // Bitwise AND
   OPdiv     = 0x1a,  // Divide
   OPjalr    = 0x09,  // Jump and link register
   OPjr      = 0x08,  // Jump register
   OPmfhi    = 0x10,  // Move from hi
   OPmflo    = 0x12,  // Move from hi
   OPmult    = 0x18,  // Multiply
   OPor      = 0x25,  // Bitwise OR
   OPsll     = 0x00,  // Shift left logical
   OPslt     = 0x2a,  // Set less than
   OPsra     = 0x03,  // Shift right arithmetic
   OPsrl     = 0x02,  // Shift right logical
   OPsub     = 0x22,  // Subtract
   OPsyscall = 0x0C,  // System call
   OPxor     = 0x26,  // Bitwise XOR
};

class Emitter : public Compiler {
public:
   Emitter();
   void openObjectFile(const string &);
   void closeObjectFile();
   void emit(IOpcode, Register rs, Register rt, long offset);
   void emit(JOpcode, long target);
   void emit(ROpcode, Register rs, Register rt, Register rd,
      long offset = 0);
   void emit(long address, long value);
   long getPC() const { return PC; }
private:
   void writeBytes(long, int);// value, byte count
   void writeT(long, long);   // write T-record to object file
   ofstream objectFile;
   long PC;                   // program counter
};

typedef Emitter *PEmitter;

#endif

