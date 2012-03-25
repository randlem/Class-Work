#include <iostream>
using std::cout;
using std::cin;
using std::endl;

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <sys/time.h>

const int TIMING_REPS = 10;
const int DATAPOINT_REPS = 5;

int ourSort(int *list, int cnt) {
	int iTemp, left=1, right=cnt-1, t, comp=0;

	do {
		for (int i=right; i >= left; i--) {
			if (list[i] < list[i-1]) {
				iTemp 		= list[i];
				list[i] 	= list[i-1];
				list[i-1] 	= iTemp;
				t = i;

			}
			comp++;
		}

		left = t+1;
		for(int i=left; i < right+1; i++) {
			if(list[i] < list[i-1]) {
				iTemp 		= list[i];
				list[i] 	= list[i-1];
				list[i-1] 	= iTemp;
				t = i;
			}
			comp++;
		}
		right = t-1;
	} while(left <= right);

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
		cout << "Usage: oursort <list size> <in file>" << endl;
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
			comp = ourSort(list,cnt);
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
