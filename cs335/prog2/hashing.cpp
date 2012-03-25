/**********************************************************
 * Program #2
 *
 * Written by Mark Randles
 * Parts written by others are credited as such.
 *
 * Class: CS335
 * Instructor: Walter Maner
 * 
 * The objective of this program is to study common hashing
 * functions.  Ten common functions will be tested aginst
 * a list of 74 keys.  We will try to minimize the number of
 * collisions for tables that are of prime sizes between 74 
 * and 197.
 *
 *********************************************************/

#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <memory.h>
#include <vector>
#include <list>

#define PJW_HASH_SHIFT 4
#define PJW_HASH_RIGHT_SHIFT 24
#define PJW_HASH_MASK 0xf0000000

using std::vector;
using std::list;

typedef vector< int >::size_type hashValue;
typedef hashValue (*functionSig)( char * );

typedef struct {
    int table_size;
    int collisions;
    int function;
} STATS;

#define KEYWORD_COUNT 74 
static char* keywords[] = { "abstract","and","and_eq","asm","auto","bitand",
  "bitor","bool","break","case","catch","char","class",
  "compl","const","const_cast","continue","default",
  "delete","do","double","dynamic_cast","else","enum",
  "explicit","extern","false","float","for","friend",
  "goto","if","inline","int","long","mutable","namespace",
  "new","not","not_eq","operator","or","or_eq","private",
  "protected","public","register","reinterpret_cast","return",
  "short","signed","sizeof","static","static_cast","struct",
  "switch","template","this","throw","true","try","typedef",
  "typeid","typename","union","unsigned","using","virtual",
  "void","volatile","wchar_t","while","xor","xor_eq" };

// Original Source: http://www.cis.temple.edu/~ingargio/cis67/code/hashtest/hashfunction.cpp
// Modification made by Mark Randles
hashValue hashfnct0(char *str) {
    hashValue hash = 5381;
    int c;
    while ( c = *str++ )
        hash = ( ( hash << 5 ) + hash ) + c; /* hash * 33 + c */
    return(hash);
}

// Original Source: http://www.cis.temple.edu/~ingargio/cis67/code/hashtest/hashfunction.cpp
// Modification made by Mark Randles
hashValue hashfnct1(char* s) {
    unsigned int const shift = 6;
    int const mask = ~0U << ( 8 * sizeof( int ) - shift );
    hashValue result = 0;
    for ( unsigned int i = 0; i < strlen(s); ++i )
        result = ( result & mask ) ^ ( result << shift ) ^ s [ i ];
    return(result);
}

// Original Source: http://www.cis.temple.edu/~ingargio/cis67/code/hashtest/hashfunction.cpp
// Modification made by Mark Randles
hashValue hashfnct2(char* s ) {
    hashValue result = 0;
    for ( unsigned int i = 0; i < strlen(s); ++i )
        result = 131 * result + s[ i ];
    return(result);
}

// Original Source: http://www.cs.uidaho.edu/~warn6645/CS213/hash.cpp
// Modification made by Mark Randles
hashValue hashfnct3(char* s ) {
    hashValue sum = 0;
    char l[strlen(s)]; //used to convert to lower case
    strcpy( l, s ); //values in order to get an accurate
    for ( int i = 0; l[ i ] != '\0'; i++ )  //index
    {
        if ( l[ i ] < 'a' || l[ i ] > 'z' )
            l[ i ] += ( 'a' - 'A' );
    }
    for ( int j = 0; l[ j ] != '\0'; j++ )
        sum += ( l[ j ] - 'a' );
    hashValue index = ( hashValue ) ldexp( sum, 18 );
    return(index);
}

// Original Source: http://src.openresources.com/debian/src/devel/HTML/S/linux86_0.13.0.orig%20linux-86%20unproto%20hash.c.html#34
// Modification made by Mark Randles
hashValue hashfnct4(char* s) {
    hashValue h = 0;
    unsigned long g;
    while ( *s )
    {
        h = ( h << 4 ) + *s++;
        if ( g = ( h & 0xf0000000 ) )
        {
            h ^= ( g >> 24 );
            h ^= g;
        }
    }
    return(h);
}

// Original Source: http://csweb.cs.bgsu.edu/maner/335/hashfns2.cpp.HTML
// Modification made by Mark Randles
hashValue hashfnct5(char* key)
{
    hashValue hash = 0;
    for(int i=0; i < strlen(key); i++) {
        hash = (hash << PJW_HASH_SHIFT) + key[i];
        unsigned int rotate_bits = hash & PJW_HASH_MASK;
        hash ^= rotate_bits | (rotate_bits >> PJW_HASH_RIGHT_SHIFT);
    }
    return(hash);
}

// Original Source: http://joda.cis.temple.edu/~wolfgang/cis542/Week11.pdf
// Modifications made by Mark Randles
hashValue hashfnct6(char* key) {
    hashValue res = 0;
    unsigned s = sizeof( hashValue ) * 8 - 1;
    const char*p = key;
    while ( *p )
        res = (res<<1) ^ *p++;
    return res;
}


// Created by Mark Randles
hashValue hashfnct7(char* key) {
    int i;
    hashValue hash;
    for(i=0; i < strlen(key); i++) {
        if(i%2 == 0) 
            hash += key[i];
        else
            hash -= key[i];
    }
    if(hash > 0) return(hash); else return(-hash);
}

// Created by Mark Randles
hashValue hashfnct8(char* key) {
    int i;
    hashValue hash;
    for(i=0; i < strlen(key); i++) {
        hash += (key[i] << i) % strlen(key);
    }
    return(hash);
}

// Original Source: http://burtleburtle.net/bob/hash/doobs.html
// Modifications made by Mark Randles
hashValue hashfnct9(char *key) {
    hashValue hash;
    int i;
    for(hash=0, i=0; i<strlen(key); ++i)
    {
        hash += key[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return (hash);
}

// taken from project website, written by Walter Maner
int nextPrime( int N ) {	
    int i;	
    if( N % 2 == 0 )	
        N++;	
    else	
      N += 2;	
    for( ; ; N += 2 ) {	
        for( i = 3; i * i <= N; i +=2 )	
            if( N % i == 0 )	
                break;	
        if( i * i > N )	
            return N;	
    }	
}

int processHashTable(int *hash_table,int table_size) {
    int i,collisions=0;

    for(i=0; i < table_size; i++) {
        if(hash_table[i] > 1)
            collisions += hash_table[i] - 1;
    }
    
    return(collisions);
}

// globally avaliable function array of the hash functions
#define NUMB_HASH_FNCT 10
functionSig hashFnct[] = { hashfnct0,hashfnct1,hashfnct2,hashfnct3,hashfnct4,hashfnct5,hashfnct6,hashfnct7,hashfnct8,hashfnct9 };

int main() {
    int i,j,*hash_table=NULL,table_size=0;
    hashValue hash;
    list< STATS > fnct_stats;
    STATS temp;
        
    for(i=0; i < NUMB_HASH_FNCT; i++) {
        list< STATS > stats;
        printf("\nFUNCTION #%i\n",i);
        printf("        Table Size        # of Colisions\n");
        printf("    ----------------------------------------\n");
        for(table_size = nextPrime(74); table_size < 199; table_size = nextPrime(table_size)) {
            if(hash_table != NULL) { delete[] hash_table; }
            hash_table = new int[table_size];
            memset(hash_table,0,sizeof(hashValue)*table_size);
            
            for(j=0; j < KEYWORD_COUNT; j++) {
                hash = hashFnct[i](keywords[j]);
                hash_table[(int)(hash%table_size)]++;
                //printf("%s %i %i\n",keywords[j],(int)(hash%table_size),hash_table[(int)(hash%table_size)]);
            }
            temp.table_size = table_size;
            temp.collisions = processHashTable(hash_table,table_size);
            stats.push_back(temp);
        }
        temp.collisions = (*stats.begin()).collisions;
        for(list<STATS>::iterator iter=stats.begin(); iter!=stats.end(); ++iter) {
            if((*iter).collisions < temp.collisions) { temp.collisions = (*iter).collisions; temp.table_size = (*iter).table_size;}
        }
        for(list<STATS>::iterator iter=stats.begin(); iter!=stats.end(); ++iter) {
            printf("      %c    %.3i                  %i\n",((*iter).collisions == temp.collisions) ? '*' : ' ',(*iter).table_size,(*iter).collisions);
        }
        temp.function = i;
        fnct_stats.push_back(temp);
    }
    
    for(list<STATS>::iterator iter=fnct_stats.begin(); iter!=fnct_stats.end(); ++iter) {
        float per = ((float)(*iter).collisions)/((float)(*iter).table_size)*100.0F;
        printf("\nFunction #%i:\n\tTable Size = %i\n\t# of Collisions = %i\n\t%% of Collisions = %.4f\n",(*iter).function,(*iter).table_size,(*iter).collisions,per);
    }
    
    exit(0);
}

/*
FUNCTION #0
        Table Size        # of Colisions
    ----------------------------------------
           079                  24
           083                  23
           089                  29
           097                  22
           101                  22
           103                  20
           107                  20
           109                  21
           113                  15
           127                  20
           131                  20
           137                  15
           139                  17
           149                  15
           151                  18
           157                  16
           163                  13
           167                  18
      *    173                  11
           179                  13
           181                  15
           191                  15
      *    193                  11
      *    197                  11

FUNCTION #1
        Table Size        # of Colisions
    ----------------------------------------
           079                  29
           083                  25
           089                  25
           097                  24
           101                  17
           103                  26
           107                  19
           109                  18
           113                  22
           127                  17
           131                  17
           137                  22
           139                  14
           149                  20
           151                  14
           157                  13
           163                  15
           167                  14
           173                  17
           179                  16
      *    181                  8
           191                  18
           193                  14
           197                  14

FUNCTION #2
        Table Size        # of Colisions
    ----------------------------------------
           079                  23
           083                  25
           089                  24
           097                  20
           101                  19
           103                  20
           107                  21
           109                  21
           113                  21
           127                  21
           131                  28
           137                  20
           139                  20
           149                  13
           151                  13
      *    157                  11
           163                  13
           167                  14
           173                  16
           179                  13
           181                  15
           191                  17
      *    193                  11
           197                  12

FUNCTION #3
        Table Size        # of Colisions
    ----------------------------------------
           079                  27
           083                  28
           089                  25
           097                  25
           101                  25
           103                  25
           107                  25
           109                  26
           113                  25
           127                  25
           131                  25
      *    137                  24
           139                  25
           149                  25
      *    151                  24
           157                  25
      *    163                  24
           167                  25
      *    173                  24
      *    179                  24
      *    181                  24
      *    191                  24
           193                  25
      *    197                  24

FUNCTION #4
        Table Size        # of Colisions
    ----------------------------------------
           079                  28
           083                  22
           089                  26
           097                  24
           101                  19
           103                  22
           107                  21
           109                  21
           113                  17
           127                  18
           131                  17
      *    137                  10
           139                  19
           149                  16
           151                  12
           157                  18
      *    163                  10
           167                  17
           173                  16
           179                  19
           181                  12
           191                  13
      *    193                  10
           197                  11

FUNCTION #5
        Table Size        # of Colisions
    ----------------------------------------
           079                  28
           083                  22
           089                  26
           097                  24
           101                  19
           103                  22
           107                  21
           109                  21
           113                  17
           127                  18
           131                  17
      *    137                  10
           139                  19
           149                  16
           151                  12
           157                  18
      *    163                  10
           167                  17
           173                  16
           179                  19
           181                  12
           191                  13
      *    193                  10
           197                  11

FUNCTION #6
        Table Size        # of Colisions
    ----------------------------------------
           079                  25
           083                  31
           089                  24
           097                  23
           101                  21
           103                  25
           107                  15
           109                  13
           113                  18
           127                  16
           131                  22
           137                  16
           139                  12
           149                  20
           151                  13
           157                  10
           163                  12
           167                  17
           173                  12
           179                  11
           181                  15
           191                  14
           193                  16
      *    197                  9

FUNCTION #7
        Table Size        # of Colisions
    ----------------------------------------
           079                  30
           083                  23
           089                  22
           097                  23
           101                  22
           103                  19
           107                  25
           109                  22
           113                  22
           127                  18
           131                  15
           137                  18
           139                  22
           149                  21
           151                  18
           157                  14
           163                  16
           167                  10
      *    173                  9
           179                  15
           181                  13
           191                  14
           193                  10
           197                  13

FUNCTION #8
        Table Size        # of Colisions
    ----------------------------------------
           079                  28
           083                  23
           089                  26
           097                  22
           101                  24
           103                  24
           107                  20
           109                  24
           113                  23
           127                  22
           131                  17
           137                  19
           139                  19
           149                  12
           151                  16
           157                  21
      *    163                  11
           167                  16
           173                  12
           179                  14
           181                  12
           191                  14
           193                  15
           197                  18

FUNCTION #9
        Table Size        # of Colisions
    ----------------------------------------
           079                  27
           083                  24
           089                  25
           097                  23
           101                  16
           103                  20
           107                  17
           109                  27
           113                  21
           127                  15
           131                  17
           137                  20
      *    139                  7
           149                  13
           151                  23
           157                  14
           163                  15
           167                  13
           173                  14
           179                  15
           181                  12
           191                  11
           193                  16
           197                  10

Function #0:
	Table Size = 173
	# of Collisions = 11
	% of Collisions = 6.3584

Function #1:
	Table Size = 181
	# of Collisions = 8
	% of Collisions = 4.4199

Function #2:
	Table Size = 157
	# of Collisions = 11
	% of Collisions = 7.0064

Function #3:
	Table Size = 137
	# of Collisions = 24
	% of Collisions = 17.5182

Function #4:
	Table Size = 137
	# of Collisions = 10
	% of Collisions = 7.2993

Function #5:
	Table Size = 137
	# of Collisions = 10
	% of Collisions = 7.2993

Function #6:
	Table Size = 197
	# of Collisions = 9
	% of Collisions = 4.5685

Function #7:
	Table Size = 173
	# of Collisions = 9
	% of Collisions = 5.2023

Function #8:
	Table Size = 163
	# of Collisions = 11
	% of Collisions = 6.7485

Function #9:
	Table Size = 139
	# of Collisions = 7
	% of Collisions = 5.0360
    
*/
