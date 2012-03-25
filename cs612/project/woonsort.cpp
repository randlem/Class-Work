#include <iostream>
using std::cout;
using std::cin;
using std::endl;

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <vector>
using std::vector;

#include <sys/time.h>

const int TIMING_REPS = 10; // 10
const int DATAPOINT_REPS = 5; // 5

int woonSort(int *list, int cnt) {
	vector<int> breaks;
	vector<int>::iterator iter;
	int i, j, k, center, left, right, *temp, merge_iters;

	for (i=1; i < cnt; i++) {
		if (list[i] < list[i-1]) {
			breaks.push_back(i);
		}
	}
	breaks.push_back(cnt);

	if (breaks.size() == 1)
		return 0;

	temp = new int[cnt];

	merge_iters = 0;
	iter = breaks.begin();
	right = *(iter++);
	do {
		memset(temp,-1,sizeof(int)*cnt);

		left = right;
		right = *(iter++);

		for (i=0, j=left, k=0; i < left && j < right; k++) {
			if (list[i] < list[j])
				temp[k] = list[i++];
			else
				temp[k] = list[j++];
			merge_iters++;
		}

		while (i < left) {
			temp[k++] = list[i++];
			merge_iters++;
		}


		while (j < right) {
			temp[k++] = list[j++];
			merge_iters++;
		}

		for (i=0; i < right; i++)
			list[i] = temp[i];

	} while (iter != breaks.end());

	delete temp;

	return merge_iters;
}

suseconds_t timeDiff(timeval *t1, timeval *t2) {
	return (t1->tv_sec * 1000000 + t1->tv_usec) -
		(t2->tv_sec * 1000000 + t2->tv_usec);
}

int main(int argc, char* argv[]) {
	ifstream unsorted_file;
	ofstream sorted_file;
	int cnt, *unsorted, *list, comp;
	struct timeval start, end;
	suseconds_t elapsed = 0, single;

	if (argc != 4) {
		cout << "Usage: woonsort <list size> <in file> <out file>" << endl;
		return -1;
	}

	cnt = atoi(argv[1]);
	unsorted_file.open(argv[2]);
	sorted_file.open(argv[3]);

	unsorted = new int[cnt];
	for (int i=0; i < cnt || unsorted_file.eof(); i++)
		unsorted_file >> unsorted[i];

	list = new int[cnt];
	for (int i=0; i < TIMING_REPS; i++) {
		gettimeofday(&start,0);
		for (int j=0; j < DATAPOINT_REPS; j++) {
			memcpy(list,unsorted,sizeof(int)*cnt);
			comp = woonSort(list,cnt);
		}
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

	for (int i=0; i < cnt; i++)
		sorted_file << list[i] << endl;

	sorted_file.close();
	unsorted_file.close();

	delete unsorted;
	delete list;

	cout << "Time=" << (elapsed) / TIMING_REPS << endl << "Comparisons=" << comp << endl;

	return 0;
}
