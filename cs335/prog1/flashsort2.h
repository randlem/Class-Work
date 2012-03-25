void insertionsort( int array[], int arraysize ) {	
    int pos;	
    int lowest = 0; // Fixed 11/04 WM	
    for ( int pass = 1; pass < arraysize; pass++ ) {	
        pos = pass - 1;	
        lowest = array[ pass ];	
        while ( lowest < array[ pos ] && pos >= 0 ) {	
            array[ pos + 1 ] = array[ pos ];	
            --pos;	
        }	
        array[ pos + 1 ] = lowest;	
    }	
}	

// The flashsort algorithm is attributed to Karl-Dietrich Neubert
// The translation to C++ is provided by Clint Jed Casper
// Cleaning and rearranging of the code done by Mark Randles.
template<class T>
void flashsort(T* array, int length) {
	if(length == 0) return;

	int m = (int)((0.2 * length) + 2);
	
	T min, max, maxIndex;
	min = max = array[0];
	maxIndex = 0;

	for(int i = 1; i < length - 1; i += 2) {
		T small;
		T big;
		T bigIndex;

		if(array+i < array+i+1) {
			small = *(array+i);
			big = *(array+i+1);
			bigIndex = i+1;
		} else {
			big = *(array+i);
			bigIndex = i;
			small = *(array+i+1);
		}

		if(big > max) {
			max = big;
			maxIndex = bigIndex;
		}

		if(small < min) {
			min = small;
		}
	}

	if(*(array+length-1) < min) {
		min = *(array+length-1);
	}
	else if(*(array+length-1) > max) {
		max = *(array+length-1);
		maxIndex = length-1;
	}

	if(max == min) {
		return;
	}

	T* L = new T[m+1];
	
	for(int t = 1; t <= m; t++) {
		L[t] = 0;
	}
	
	double c = (m-1.0)/(max-min);
	T K;
	for(int h = 0; h < length; h++) {
		K = ((int)((*(array+h)-min)*c))+1;
		L[K] += 1;
	}
	
	for(K = 2; K <= m; K++) {
		L[K] = L[K] + L[K-1];
	}

	T temp = *(array+maxIndex);
	*(array+maxIndex) = *array;
	*array = temp;

	int j = 0;
	
	K = m;
	
	int numMoves = 0;
	
	while(numMoves < length) {
		while(j >= L[K]) {
			j++;
			K = ((int)((*(array+j) - min) * c)) + 1;
		}

		T evicted = *(array+j);

		while(j < L[K])	{
			K = ((int)((evicted-min)*c))+1;

			int location = L[K] - 1;

			T temp = *(array+location);
			*(array+location) = evicted;
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
			flashsort(array+L[K], classSize);
		} else {
			if(classSize > 1) {
				insertionsort(&array[L[K]], classSize);
			}
		}
	}

	delete [] L;
}
