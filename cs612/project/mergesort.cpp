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

int insertionSort(int* list, int cnt) {
	int val = -1, i, j, comp=0;

	for (i=1; i < cnt; i++) {
		val = list[i];
		j = i-1;
		while (j >= 0 && list[j] > val) {
			list[j+1] = list[j];
			j--;
		}
		list[j+1] = val;
		comp++;
	}
}

int mergeSort(int *list, int cnt) {
	int center, *temp, i, j, k, comp=0;

	if (cnt <= 16)
		return insertionSort(list,cnt);

	center = (int)floor(cnt / 2);

	comp += mergeSort(list,center);
	comp += mergeSort(list+center,cnt-center);

	temp = new int[cnt];
	for (i=0, j=center, k=0; i < center && j < cnt; k++) {
		if (list[i] < list[j])
			temp[k] = list[i++];
		else
			temp[k] = list[j++];
		comp++;
	}

	while (i < center)
		temp[k++] = list[i++];

	while (j < cnt)
		temp[k++] = list[j++];

	for (i=0; i < cnt; i++)
		list[i] = temp[i];

	delete temp;

	return comp;
}

suseconds_t timeDiff(timeval *t1, timeval *t2) {
	return (t1->tv_sec * 1000000 + t1->tv_usec) -
		(t2->tv_sec * 1000000 + t2->tv_usec);
}

int main(int argc, char* argv[]) {
	ifstream unsorted_file;
	int cnt, *unsorted, *list, comp;
	struct timeval start, end;
	suseconds_t elapsed = 0, single;

	if (argc != 3) {
		cout << "Usage: bubblesort <list size> <in file>" << endl;
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
			memcpy(list,unsorted,sizeof(int)*cnt);
			comp = mergeSort(list,cnt);
		}
		delete list;
		gettimeofday(&end,0);
		single = timeDiff(&end, &start) / DATAPOINT_REPS;
		if (single < 0) {
			cout << "Bad timing info" << endl;
			i--;
		} else {
			elapsed += single;
			cout << i << " " << single << endl;
		}
	}

	cout << "Time=" << (elapsed) / TIMING_REPS << endl << "Comparisons=" << comp << endl;

	return 0;
}
