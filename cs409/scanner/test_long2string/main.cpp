#include <iostream>
#include <string>
using std::string;
using std::cout;
using std::endl;

int main(int argc, char* argv[]) {

	unsigned long l = 0xFF00AB10;
	string s;

	while(l > 0) {
		unsigned char r = l % 16;
		char c = (r < 0xA) ? '0' + r : 'A' + r - 0xA;
		l /= 16;
		s = c + s;
	}

	cout << s << endl;

	return(0);
}
