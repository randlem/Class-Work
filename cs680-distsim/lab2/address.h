#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <vector>
using std::vector;

#include "base.h"
#include "simplus/UniformDist.h"

#ifndef ADDRESS_H
#define ADDRESS_H

class AddressRegister {
public:
	AddressRegister() {
		rngUniform = NULL;
	}

	~AddressRegister() {
		if(rngUniform != NULL)
			delete rngUniform;
	}

	bool registerAddress(address& a) {
		addressList.push_back(a);

		if(rngUniform != NULL) {
			delete rngUniform;
			rngUniform = NULL;
		}

		return true;
	}

	address randomAddress() {
		if(rngUniform == NULL)
			rngUniform = new UniformDist(0,addressList.size());

		return addressList[rngUniform->getRandom()];
		//return addressList[1];
	}

	int addressCount() {
		return addressList.size();
	}

private:
	vector<address> addressList;
	UniformDist* 	rngUniform;

};

#endif
