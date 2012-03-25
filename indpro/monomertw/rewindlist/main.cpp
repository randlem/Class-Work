#include <iostream>
using std::cout;
using std::endl;

#include "rewindlist.h"

int main() {
	RewindList<int> rl;
	int i;

	for(i=0; i < 10; ++i)
		rl.add(i,i);

	for(i=0; i < rl.size(); ++i)
		cout << rl[i] << endl;
	cout << "--------------------------------------" << endl;

	rl.remove(0,10);
	rl.remove(1,11);

	for(i=0; i < 10; ++i)
		rl.add(i+10,i+10);

	for(i=0; i < rl.size(); ++i)
		cout << rl[i] << endl;
	cout << "--------------------------------------" << endl;

	rl.rollback(10);

	for(i=0; i < rl.size(); ++i)
		cout << rl[i] << endl;
	cout << "--------------------------------------" << endl;


	return(0);
}
