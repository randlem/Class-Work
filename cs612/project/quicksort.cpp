#include <iostream>
using std::cout;
using std::cin;
using std::endl;

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <sys/time.h>
#include <math.h>

const int TIMING_REPS = 10;
const int DATAPOINT_REPS = 5;

int quickSort(int *data, int low, int high){
	int i, j, pivot, comp=0;

	if (low < high){
		pivot 	= data[(int)floor((low + high) / 2)];
		i 		= low;
		j 		= high;

		while (i < j){
			while(i < j && data[j] >= pivot)
				j--;

			if(i < j)
				data[i++]=data[j];

			while(i < j && data[i] <= pivot)
				i++;

			if(i < j)
				data[j--] = data[i];

			comp+=2;
		}

		data[i] = pivot;
		comp 	+= quickSort(data,low,i-1);
		comp 	+= quickSort(data,i+1,high);
	}

	return comp;
}


suseconds_t timeDiff(timeval *t1, timeval *t2) {
	return (t1->tv_sec * 1000000 + t1->tv_usec) -
		(t2->tv_sec * 1000000 + t2->tv_usec);
}

int main(int argc, char* argv[]) {
	ifstream unsorted_file;
	int *unsorted, *list, cnt, comp;
	struct timeval start, end;
	suseconds_t elapsed = 0, single;

	if (argc != 3) {
		cout << "Usage: quicksort <list size> <in file>" << endl;
		return -1;
	}

	cnt = atoi(argv[1]);
	unsorted_file.open(argv[2]);

	unsorted = new int[cnt];
	for (int i=0; i < cnt || unsorted_file.eof(); i++)
		unsorted_file >> unsorted[i];

	for (int i=0; i < TIMING_REPS; i++) {
		gettimeofday(&start,0);
		list = new int[cnt];
		for (int j=0; j < DATAPOINT_REPS; j++) {
			memcpy(list,unsorted,sizeof(int) * cnt);
			comp = quickSort(list,0,cnt-1);
		}
		delete list;
		gettimeofday(&end,0);
		single = timeDiff(&end, &start) / DATAPOINT_REPS;
		if (single < 0) {
			cout << "Bad timing info" << endl;
			i--;
		}
		else {
			elapsed += single;
			cout << i << " " << single << endl;
		}
	}

	cout << "Time=" << (elapsed) / TIMING_REPS << endl << "Comparisons=" << comp << endl;

	return 0;
}



