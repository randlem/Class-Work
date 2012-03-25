void insertionsort( int array[], int arraysize ) {	
    int pos;	
    int lowest = 0; // Fixed 11/04 WM	
    for ( int pass = 1; pass < arraysize; pass++ ) {	
        pos = pass - 1;	
        lowest = array[ pass ];	
        while ( lowest < array[ pos ] &amp;&amp; pos >= 0 ) {	
            array[ pos + 1 ] = array[ pos ];	
            --pos;	
        }	
        array[ pos + 1 ] = lowest;	
    }	
}	

template < class RandomAccessIterator >	
void isort5( RandomAccessIterator begin, RandomAccessIterator end ) {	
    for ( RandomAccessIterator outerpos = begin; 	
          outerpos != end; 	
          ++outerpos ) {	
        iterator_traits< RandomAccessIterator >::value_type	
           candidate = *outerpos;	
        RandomAccessIterator pos = outerpos - 1;	
        while ( pos >= begin &amp;&amp; candidate < *pos ) {	
            *( pos + 1 ) = *pos;	
            pos--;	
        }	
        *( pos + 1 ) = candidate;	
    }	
}	
