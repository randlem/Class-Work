#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <png.h>

typedef struct {
	FILE* fp;
	png_structp png_pt;
} PNG_FILE;


int create_file(PNG_FILE* png_file);
int close_file(PNG_FILE* png_file);
int write_file(PNG_FILE* png_file, void* 