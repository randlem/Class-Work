#include <stdio.h>
#include <stdlib.h>

int main() {
	int input,i;

    i=0;
    while(scanf("%i\n",&input) != EOF) {
        if(input != i)
            printf("%i %i\n",i,input);
        i++;
    }
    printf("total = %i\n",i);
} 
