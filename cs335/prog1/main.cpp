#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include "Timer.h"
#include "flashsort4.h"

#define N 100000000

using std::vector;	
using std::random_shuffle;	

int main(int argv, char* argc[]) {
    vector<int> vi;	
    for (int i = 0; i < N; i++)
        vi.push_back( i );
        
    random_shuffle(vi.begin(), vi.end());
	
    //for(vector<int>::iterator i = vi.begin(); i != vi.end(); ++i)
    //    printf("%i\n",*i);
    
    flashsort(vi.begin(), vi.end());
	
    //for(vector<int>::iterator i = vi.begin(); i != vi.end(); ++i)
    //    printf("%i\n",*i);

    if(is_sorted(vi.begin(),vi.end())) printf("sorted"); else printf("not sorted");
        
	return(0);

}// end main()
