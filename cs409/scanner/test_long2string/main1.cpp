#include <iostream>
#include <string>
using std::string;
using std::cout;
using std::endl;

int main(int argc, char* argv[]) {

	long l = 0xFFFFFFFF;
	string s;

	/*for(int i=0; i < 8; i++) {
		char c = 0x0000000F & (l >> (i * 4));
		s += ((c < 0xA) ? '0' + c : 'A' + c - 0xA);
	}*/

	for(int i=7; i >= 0; i--) {
		char c = 0x0000000F & (l >> (i * 4));
		s += ((c < 0xA) ? '0' + c : 'A' + c - 0xA);
	}

	cout << s << endl;

	return(0);
}
