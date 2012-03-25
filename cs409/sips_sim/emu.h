#include <stdio.h>

#ifndef __EMU_H__
#define __EMU_H__

#define EMU_ERROR_INVALID_EOF    0x00000001
#define EMU_ERROR_MALFORMED_OBJ  0x00000002
#define EMU_ERROR_CORRUPT_HEADER 0x00000004

#define ADDRESS_SPACE 0x080000
#define TEXT_SEGMENT  0x040000
#define DATA_SEGMENT  0x044000
#define STACK_START   0x048000

#define MIN_INPUT_SIZE  0x5
#define OBJ_FILE_HEADER 0xBEAD4509
#define OBJ_FILE_DELIM  0x54
#define OBJ_FILE_EOF    0x45

typedef unsigned int sips_word;
typedef sips_word sips_register;

typedef struct {
	sips_word address_space[ADDRESS_SPACE];
	const sips_register zero;
	sips_register v0;
	sips_register v1;
	sips_register a0;
	sips_register a1;
	sips_register a2;
	sips_register a3;
	sips_register t0;
	sips_register t1;
	sips_register t2;
	sips_register t3;
	sips_register s0;
	sips_register s1;
	sips_register s2;
	sips_register s3;
	sips_register gp;
	sips_register sp;
	sips_register fp;
	sips_register ra;
} emu;

void init_emu(emu* emulator);

int load_emu(emu* emulator, FILE* input_file);

#endif
