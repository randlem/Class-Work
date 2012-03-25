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

void countingSort(int* unsorted, int *sorted, int cnt, const int max) {
	int counts[100000];

	memset(counts,0,sizeof(int)*100000);

	for (int i=0; i < cnt; i++)
		counts[unsorted[i]]++;

	for (int i=1; i < max; i++)
		counts[i] += counts[i-1];

	for (int i=cnt; i >= 0; i--) {
		sorted[counts[unsorted[i]]-1] = unsorted[i];
		counts[unsorted[i]]--;
	}
}

suseconds_t timeDiff(timeval *t1, timeval *t2) {
	return (t1->tv_sec * 1000000 + t1->tv_usec) -
		(t2->tv_sec * 1000000 + t2->tv_usec);
}

int main(int argc, char* argv[]) {
	ifstream unsorted_file;
	ofstream sorted_file;
	int cnt, *list, *sorted, max;
	struct timeval start, end;
	suseconds_t elapsed = 0, single;

	if (argc != 4) {
		cout << "Usage: bubblesort <list size> <in file> <out file>" << endl;
		return -1;
	}

	cnt = atoi(argv[1]);
	unsorted_file.open(argv[2]);
	sorted_file.open(argv[3]);

	list = new int[cnt];
	sorted = new int[cnt];

	max = 0;
	for (int i=0; i < cnt || unsorted_file.eof(); i++) {
		unsorted_file >> list[i];
		if (max < list[i])
			max = list[i];
	}

	for (int i=0; i < TIMING_REPS; i++) {
		gettimeofday(&start,0);
		for (int j=0; j < DATAPOINT_REPS; j++)
			countingSort(list,sorted,cnt,max);
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

	for (int i=0; i < cnt; i++)
		sorted_file << sorted[i] << endl;

	cout << (elapsed) / TIMING_REPS << "us on average." << endl;

	return 0;
}
