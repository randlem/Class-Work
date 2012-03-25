#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include "emu.h"

void init_emu(emu* emulator) {
	memset(emulator,0,sizeof(emu));
}

int load_emu(emu* emulator, FILE* input_file) {
	unsigned char* buffer;
	unsigned char len;
	unsigned int header = 0;

	/* get the lenght of the file */
	fseek(input_file,0,SEEK_END);
	len = ftell(input_file);
	rewind(input_file);

	/* check to make sure the header exists */
	if(len < MIN_INPUT_SIZE) {
		fprintf(stderr,"Unexpected End-of-File encountered!\n");
		return(EMU_ERROR_INVALID_EOF);
	}

	/* allocate the buffer memory */
	buffer = (unsigned char*)malloc(sizeof(unsigned char) * len);
	memset(buffer,0,sizeof(unsigned char) * len);

	/* read the entire input_file into the buffer */
	fread(buffer,sizeof(unsigned char),len,input_file);

	/* parse off the first 4 bytes */
	header = (*buffer << 12) + (*(buffer+1) << 8) + (*(buffer+2) << 4) + (*(buffer+3) << 0);
	buffer+=4;

	/* validate header */
	if(header != HEADER_SIG) {
		fprintf(stderr,"Corrupt file header signature!\n");
		return(EMU_ERROR_CORRUPT_HEADER);
	}

	/* load the file into memory */


	/* free the buffer memory */
	free(buffer);

	return(0);
}
