#include <stdio.h>
#include <stdlib.h>

#define A_ROW 2000
#define A_COL 1000
#define B_ROW 1000
#define B_COL 2500

int main() {
    int A[A_ROW][A_COL],
        B[B_ROW][B_COL],
        C[A_COL][B_ROW];
    int i,j,k,l,sum;

    /* initlization */
    for(i=0; i < A_ROW; i++)
        for(j=0; j < A_COL; j++)
            A[i][j] = i*j;
    for(i=0; i < B_ROW; i++)
        for(j=0; j < B_COL; j++)
            B[i][j] = i*j;
    memset(C,0,sizeof(int)*A_COL*B_ROW);

    for(i=0; i < A_ROW; i++) {
        for(j=0; j < B_COL; j++) {
            sum = 0;
            for(k=0; k < A_COL; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }

    return(0);
}
