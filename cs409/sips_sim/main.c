#define DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <errno.h>

#include "utility.h"
#include "emu.h"

#define NUM_OPTS    1
#define OUTPUT_FILE 0

#define IS_SET(a)         (opt_flags[a] == 1)
#define SET_FLAG(a)       (opt_flags[a] = 1)
#define UNSET_FLAG(a)     (opt_flags[a] = 0)
#define GET_FLAG(a)       (opt_flags[a])
#define GET_FLAG_PRINT(a) ((opt_flags[a] == 1) ? "TRUE" : "FALSE")

typedef unsigned char opt_type;

char*    input_file  = NULL;  /* the input file path and name */
char*    output_file = NULL;  /* the output file path and name if so set */
FILE*    in_file     = NULL;  /* file pointer to the input file */
FILE*    out_file    = NULL;  /* file pointer to the output file */
opt_type opt_flags[NUM_OPTS]; /* array of option flags (0 if unset, 1 if set) */

int parse_options(int argc, char* argv[]);
void global_cleanup();

int main(int argc, char* argv[]) {
	emu emulator;

	/* validate command line */
	if(argc >= 2) {
		if(!parse_options(argc,argv)) {
			fprintf(stderr,"%s",USAGE_MESSAGE);
			return(1);
		}
	} else {
		fprintf(stderr,"No input file specified!\n%s",USAGE_MESSAGE);
		return(1);
	}

#ifdef DEBUG
	int debug_i;
	printf("Option dump:\n");
	for(debug_i=0; debug_i < NUM_OPTS; debug_i++) {
		switch(debug_i) {
			case OUTPUT_FILE:
			{
				printf("\tOUTPUT_FILE: %s\n",GET_FLAG_PRINT(OUTPUT_FILE));
			} break;
			default:
			{
			}
		}
	}
	printf("\n");
#endif

	/* open the input file */
	if((in_file = fopen(input_file,"r")) == NULL) {
		perror("Couldn't open the input file");
		global_cleanup();
		return(1);
	}

	/* is the output file opt is set then we need to open the output file and the
	   pointer to it in the output file instead of stdout */
	out_file = stdout;
	if(IS_SET(OUTPUT_FILE)) {
		if((out_file = fopen(output_file,"w")) == NULL) {
			perror("Couldn't open the output file");
			global_cleanup();
			return(1);
		}
	}

	/* do some initial text output */
	fprintf(out_file,OUT_FILE_PREAMBLE);
	fprintf(out_file,"Using input file %s\n",input_file);
	if(IS_SET(OUTPUT_FILE)) {
		fprintf(out_file,"Using output file %s\n",output_file);
	}

	/* init the emu and load the obj file into memory */
	init_emu(&emulator);
	if(load_emu(&emulator,in_file) > 0) {
		global_cleanup();
		return(1);
	} else {
		fprintf(out_file,"Loaded object file into emulator memory.\n");
	}

	/* clean up after myself */
	global_cleanup();

	/* return to the system normally */
	return(0);
}

int parse_options(int argc, char* argv[]) {
	int i;

	if(input_file != NULL) {
		free(input_file);
		input_file = NULL;
	}
	memset(opt_flags,0,sizeof(opt_type) * NUM_OPTS);

	/* handle the special case of just the input file */
	if(argc == 2) {
		input_file = malloc(sizeof(char) * (strlen(argv[1])+1));
		strcpy(input_file, argv[1]);
		return(1);
	}

	/* handle the options then the input filename */
	for(i=1; i < argc-1; i++) {
		char* str = argv[i];

		if(*str == '-') {
			switch(*(str+1)) {
				case 'o':
				{
					SET_FLAG(OUTPUT_FILE);

					if(output_file != NULL) {
						free(output_file);
						output_file = NULL;
					}

					if(i+1 < argc-1) {
						output_file = malloc(sizeof(char) * (strlen(argv[i++])));
						strcpy(output_file,argv[i]);
					} else {
						fprintf(stderr,"Too few parameters for %s option!\n",str);
						return(0);
					}

				}break;
				default:
				{
					fprintf(stderr,"Unknown option %s\n",str);
					return(0);
				}
			}
		} else {
			fprintf(stderr,"Unknown option format!\n");
			return(0);
		}
	}
	input_file = malloc(sizeof(char) * (strlen(argv[argc-1])+1));
	strcpy(input_file, argv[argc-1]);

	return(1);
}

void global_cleanup() {
	if(input_file != NULL) {
		free(input_file);
		input_file = NULL;
	}

	if(IS_SET(OUTPUT_FILE)) {
		UNSET_FLAG(OUTPUT_FILE);
		free(output_file);
		output_file = NULL;
	}

	if(output_file != NULL) {
		free(output_file);
		output_file = NULL;
	}

	fclose(in_file);
	fclose(out_file);
}
