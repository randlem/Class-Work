#include <iostream>
using std::cout;
using std::cin;
using std::endl;

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <time.h>

const int TIMING_REPS = 100;
const int DATAPOINT_REPS = 5;

void heapify(int low,int high){
 int large;
 int temp;
 int *data;
 temp=data[low];
 for(large=2*low;large<=high;large*=2){
  if(large<high&&data[large]<data[large+1])
	  large++;
  if(temp>=data[large])break;
  data[low]=data[large];
  low=large;
 }
  data[low]=temp;
}
void buildheap(){
 int i;
 int cnt;
 for(i=cnt/2;i>=1;i--)
     heapify(i,cnt);
}
void heapSort(int *data,int cnt){
 int i;
 buildheap();
 for(i=cnt;i>1;i--){
	 data[0]=data[1];
	 data[1]=data[i];
	 data[i]=data[0];
     heapify(1,i-1);
 }
}
suseconds_t timeDiff(timeval *t1, timeval *t2) {
	return (t1->tv_sec * 1000000 + t1->tv_usec) -
		(t2->tv_sec * 1000000 + t2->tv_usec);
}

int main(int argc, char* argv[]) {
	ifstream unsorted_file;
	ofstream sorted_file;
	int cnt, *data;
	struct timeval start, end;
	suseconds_t elapsed = 0, single;

	if (argc != 4) {
		cout << "Usage: heapsort <list size> <in file> <out file>" << endl;
		return -1;
	}

	cnt = atoi(argv[1]);
	unsorted_file.open(argv[2]);
	sorted_file.open(argv[3]);

	data = new int[cnt];

	for (int i=0; i < cnt || unsorted_file.eof(); i++)
		unsorted_file >> data[i];

	for (int i=0; i < TIMING_REPS; i++) {
		gettimeofday(&start,0);
		for (int j=0; j < DATAPOINT_REPS; j++)
			heapSort(data,cnt);
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
		sorted_file << data[i] << endl;

	cout << (elapsed) / TIMING_REPS << "us on average." << endl;

	return 0;
}
