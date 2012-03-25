#include <iostream>
using std::cout;
using std::endl;

#include <queue>
using std::priority_queue;

#include "../genrand/genrand_cpp.h"

int main() {
	priority_queue<double> q;
	RandGenEngine rge;

	for(int i = 0; i < 10; ++i) {
		q.push(rge.randNum());
	}
	
	while(!q.empty()) {
		cout << q.top() << endl;
		q.pop();
	}

	return(0);
}
