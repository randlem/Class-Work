//   File:         emitter.cpp
//   Author(s):    Ron Lancaster
//
//   Contents:
//      Implementation of the Emitter class for
//      the SIPS Virtual Machine.

#include <iomanip>
using std::hex;
using std::ios;
using std::setfill;
using std::setw;

#include <iostream>
using std::endl;

#include "emitter.h"

// Declare static array

//   Emitter::Emitter()
//
//   This function initializes class data.

Emitter::Emitter() : Compiler()
{
   PC = 0x40000;          // start of object code
}

//   Emitter::closeObjectFile()
//
//   This function outputs the final 'E' record and closes the object file.

void Emitter::closeObjectFile()
{
   if(!objectFile)
      setError("Close: object file not open");
   writeBytes('E', 1);
   objectFile.close();
}

//   Emitter::emit(IOpcode, Register, Register, long)
//
//   This function generates I-Type instructions.

void Emitter::emit(IOpcode op, Register rs, Register rt, long offset)
{
   long instruction =  (op << 26)
      | (rs << 21)
      | (rt << 16)
      | (offset & 0xFFFF);
   writeT(PC, instruction);
   PC += 4;
}

//   Emitter::emit(JOpcode op, long target)
//
//   This function generates J-Type instructions.

void Emitter::emit(JOpcode op, long target)
{
   long instruction = (op << 26) | target;
   writeT(PC, instruction);
   PC += 4;
}

//   Emitter::emit(ROpcode op, Register rs, Register rt, Register rd,
//      long offset)
//
//   This function generates R-Type instructions.

void Emitter::emit(ROpcode op, Register rs, Register rt, Register rd,
   long offset)
{
   long instruction = (rs << 21)
      | (rt << 16)
      | (rd << 11)
      | ((offset & 0x1F) << 6)
      | op;
   writeT(PC, instruction);
   PC += 4;
}

//   Emitter::emit(long address, long value)
//
//   This function stores a value at a specific address in the data
//   area.  The argument must be offset by the base of the data area.

void Emitter::emit(long address, long value)
{
   writeT(0x44000 + address, value);
}

//   Emitter::openObjectFile
//
//   This function opens the object file in binary mode and then
//   outputs the header record.

void Emitter::openObjectFile(const string &name)
{
   objectFile.open(name.c_str(), ios::binary);
   if(!objectFile)
      setError(string("Couldn't open object file ") + name);
   // Write header record
   writeBytes(0xbead4509L, 4);
}

//   Emitter::writeBytes(long, int)
//
//   This function writes bytes to the object code file.

void Emitter::writeBytes(long value, int bytes)
{
   unsigned char code[8];
   int c=0;  // index for code array

   if ((bytes < 1) || (bytes > 8))
      setError(string("Invalid byte count for emit function: ")
         + long2string(bytes));

   for (int i=bytes-1; i>=0; --i) {
      code[c++] = (unsigned char)((value >> (8*i)) & 0xFF);
   }

   objectFile.write((char *) code, bytes);
}

//   Emitter::writeT(long location, long value)
//
//   This function writes a T record to the object file to save a
//   non-zero integer value.

void Emitter::writeT(long location, long value)
{
   if (!objectFile.is_open())
      setError("Object file is not open!");
   if(value != 0) {
      writeBytes('T', 1);
      writeBytes(location, 4);
      writeBytes(4, 2); // instruction length
      writeBytes(value, 4);
      // Show object code
      if(debugFlag)
         debug << setw(5) << hex << setfill('0') << location << ": "
               << setw(8) << value
               << setfill(' ') << endl;
   }
}
