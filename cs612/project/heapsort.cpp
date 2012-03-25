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

int siftDown (int *list, int start, int end) {
	int root = start, child, t, comp=0;

	while (root * 2 < end) {
		child = root * 2;

		if (child+1 <= end && list[child] < list[child + 1])
			child = child + 1;

		if (list[root] < list[child]) {
			t = list[root];
			list[root] = list[child];
			list[child] = t;
			root = child;
		} else
			break;
		comp += 2;
	}

	return comp;
}

int heapify (int *list, int cnt) {
        int start = (int)floor((cnt - 2) / 2), comp=0;

        while (start >= 0) {
                comp += siftDown(list,start,cnt-1);
                start--;
        }

	return comp;
}

int heapSort (int *list, int cnt) {
	int end = cnt - 1, t, comp=0;

	comp += heapify(list,cnt);

	while (end > 0) {
		t = list[end];
		list[end] = list[0];
		list[0] = t;
		end--;
		comp += siftDown(list, 0, end);
	}

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
			comp = heapSort(list,cnt);
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
