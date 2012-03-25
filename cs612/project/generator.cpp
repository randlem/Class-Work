#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <fstream>
using std::ofstream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <algorithm>
using std::swap;
using std::fill;

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const string USAGE_MESSAGE = "Usage: generator <listSize> <fileName> <listType> [listOpts]";
const string USAGE_MESSAGE_SORTAORDER = "Usage: generator <listSize> <fileName> sortaorder <number_of_inversions>";
const string USAGE_MESSAGE_REVSORTAORDER = "Usage: generator <listSize> <fileName> revsortaorder <number_of_inversions>";

const int MIN_NUMBER = 0;
const int MAX_NUMBER = 99999;

ofstream sorted_list;
ofstream unsorted_list;

// generate a random in a range, defaults to [MIN_NUMBER,MAX_NUMBER]
int random_int (int min = MIN_NUMBER,int max = MAX_NUMBER) {
	return (int)((((float)rand()/RAND_MAX) * max) + min);
}

// genearate listSize random numbers and output to stdout
void generate_random (int listSize) {
	vector<int> numbers(MAX_NUMBER);
	int ri;
	int null_field = MIN_NUMBER - 1;

	fill(numbers.begin(), numbers.end(), null_field);

	for (int i=0; i < listSize; i++) {
		do {
			ri = random_int();
		} while (numbers[ri] != null_field);
		unsorted_list << ri << endl;
		numbers[ri] = ri;
	}

	for (vector<int>::iterator i=numbers.begin(); i != numbers.end(); ++i)
		if ((*i) != null_field)
			sorted_list << (*i) << endl;
}

// generate a numerically sequential list ascending or descending
void generate_inorder (int listSize, bool reverse = false) {
	for(int i=1; i <= listSize; ++i) {
		unsorted_list << ((reverse) ? listSize - i : i) << endl;
		sorted_list << i << endl;
	}
}

// generate a list with random inversions
void generate_sortaorder (int listSize, int inversions = 0, bool reverse = false) {
	vector<int> list(listSize);
	vector<int>::iterator iter;
	int li1 = 0;
	int li2 = 0;

	// default to 10% of the list is inverted
	if (inversions <= 0)
		inversions = (int)(listSize * 0.10);

	// generate the in-order list
	for (int i=1; i <= listSize; ++i) {
		list[i-1] = ((reverse) ? listSize - i : i);
		sorted_list << i << endl;
	}

	// randomly invert a number of list items
	for (int i=0; i < inversions; ++i) {
		li1 = random_int(0,listSize - 1);
		while ((li2 = random_int(0,listSize - 1)) == li1) { ; }
		swap(list[li1],list[li2]);
	}

	// output the randomly inverted list
	for (iter = list.begin(); iter != list.end(); ++iter)
		unsorted_list << (*iter) << endl;

}

int main (int argc, char* argv[]) {
	int listSize = 0;
	int inversions = 0;
	string filename = "list";
	string sort_type = "";

	if (argc < 3) {
		cerr << USAGE_MESSAGE << endl;
		return -1;
	}

	// get the list size
	listSize = atoi(argv[1]);

	if (listSize <= 0) {
		cerr << "List size must be larger then zero." << endl;
		return -1;
	}

	// get the base filename
	filename = argv[2];

	// seed the random number generator
	srand(time(NULL));

	// open up the sorted and unsorted file streams
	sorted_list.open((filename + ".sorted").c_str());
	unsorted_list.open((filename + ".unsorted").c_str());

	// process rest of the command line based on the list type
	sort_type = argv[3];
	if (sort_type == "random") {  // generate a random list
		cout << "Generating a random list." << endl;
		generate_random(listSize);

	} else if (sort_type == "inorder") { // generate a list in order
		cout << "Generating an in-order list." << endl;
		generate_inorder(listSize);

	} else if (sort_type == "sortaorder") { // generate a list that's mostly in order
		if (argc != 5) {
			cerr << USAGE_MESSAGE_SORTAORDER << endl;
			return -1;
		}

		cout << "Generating a sorted list with random inversions." << endl;
		inversions = atoi(argv[4]);

		generate_sortaorder(listSize,inversions);

	} else if (sort_type == "revorder") { // generate a list in reverse order
		cout << "Generating a reverse in-order list." << endl;
		generate_inorder(listSize,true);

	} else if (sort_type == "revsortaorder") { // generate a list that's mostly in reverse order
		if (argc != 5) {
			cerr << USAGE_MESSAGE_REVSORTAORDER << endl;
			return -1;
		}

		cout << "Generating a reverse sorted list with random inversions." << endl;
		inversions = atoi(argv[4]);
		generate_sortaorder(listSize,inversions,true);

	} else {
		cerr << "Couldn't understand list type.  Possible values are "
			 << "random,inorder,sortaorder,revorder,revsortaorder." << endl;
		return -2;
	}

	// close the file streams
	sorted_list.close();
	unsorted_list.close();

	return 0;
}
