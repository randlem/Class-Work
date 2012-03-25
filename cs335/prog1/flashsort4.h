#include <vector>
#include <iterator>
#include <stdlib.h>
#include <iostream.h>

template < class RandomAccessIterator >	
void isort5( RandomAccessIterator begin, 	
             RandomAccessIterator end ) {
    /*printf("insertion_sort\n");	
    for(RandomAccessIterator err=begin; err!=end; ++err) {
	printf("%i\n",*err);
    } printf("\n");fflush(stdout);
    */
    for ( RandomAccessIterator outerpos = begin; 	
          outerpos != end; 	
          ++outerpos ) {	
        iterator_traits< RandomAccessIterator >::value_type	
           candidate = *outerpos;	
        RandomAccessIterator pos = outerpos - 1;	
        while ( pos >= begin && candidate < *pos ) {	
            *( pos + 1 ) = *pos;	
            pos--;	
        }	
        *( pos + 1 ) = candidate;	
    }	
}	

// The flashsort algorithm is attributed to Karl-Dietrich Neubert
// The translation to C++ is provided by Clint Jed Casper
// Cleaning, rearranging, and conversion by Mark Randles
template < class RandomAccessIterator >
void flashsort(RandomAccessIterator begin, RandomAccessIterator end) {
    typedef iterator_traits< RandomAccessIterator >::value_type T;

	if(begin == end) return;

	printf("flashsort\n");fflush(stdout);
	/*for(RandomAccessIterator err=begin; err != end; ++err) {
		printf("%i\n",*err);
	} printf("\n");fflush(stdout);
        */
	int length = end - begin;

	int m = (int)((0.2 * length) + 2);
	
	T min, max, maxIndex;
	min = max = *begin;
	maxIndex = 0;

	for(int i = 1; i < length - 1; i += 2) {
		T small, big, bigIndex;

		if(*(begin+i) < *(begin+i+1)) {
			small = *(begin+i);
			big = *(begin+i+1);
			bigIndex = i+1;
		} else {
			big = *(begin+i);
			bigIndex = i;
			small = *(begin+i+1);
		}

		if(big > max) {
			max = big;
			maxIndex = bigIndex;
		}

		if(small < min) min = small;
	}

	if(*(begin+length-1) < min) {
                min = *(begin+length-1);
        }
	else if(*(begin+length-1) > max) {
		max = *(begin+length-1);
		maxIndex = length-1;
	}

	if(max == min) return;

	T* L = new T[m+1];
	
	for(int t = 1; t <= m; t++) L[t] = 0;
	
	double c = (m-1.0)/(max-min);
	int K;
	for(RandomAccessIterator h=begin; h != end; ++h) {
		K = ((int)(((*h)-min)*c))+1;
    	        L[K] += 1;
	}
	
	for(K = 2; K <= m; K++) {
		L[K] = L[K] + L[K-1];
	}

	T temp = *(begin+maxIndex);
	*(begin+maxIndex) = *begin;
	*begin = temp;

	int j = 0;
	K = m;
	int numMoves = 0;
	
	while(numMoves < length) {
		while(j >= L[K]) {
			j++;
			K = ((int)((*(begin+j) - min) * c)) + 1;
		}

		T evicted = *(begin+j);

		while(j < L[K])	{
			K = ((int)((evicted-min)*c))+1;

			int location = L[K] - 1;

			T temp = *(begin+location);
			*(begin+location) = evicted;
			evicted = temp;

			L[K] -= 1;

			numMoves++;
		}
	}

	int threshold = (int)(1.25*((length/m)+1));
	const int minElements = 30;
	
	for(K = m - 1; K >= 1; K--) {
		int classSize = L[K+1] - L[K];

		if(classSize > threshold && classSize > minElements) {
			//flashsort(array+L[K], classSize);
			printf("recurse");fflush(stdout);
			flashsort(begin+L[K],begin+L[K]+classSize);
		} else {
			if(classSize > 1) {
				//insertionsort(&array[L[K]], classSize);
                                isort5(begin+L[K],begin+L[K]+classSize);
			}
		}
	}

	delete [] L;
}
