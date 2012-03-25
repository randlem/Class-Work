#include <string>
using std::string;

#ifndef __HISTOGRAM_H__
#define __HISTOGRAM_H__

class Histogram {
public:
	Histogram(int cnt, int init=0) {
		this->cnt = cnt;
		this->bins = new int[this->cnt];
		this->clear(init);
	}

	Histogram(const Histogram & copy) {
		this->cnt = copy.cnt;
		this->bins = new int[this->cnt];
		memcpy(this->bins,copy.bins,sizeof(int) * this->cnt);
	}

	~Histogram() {
		delete [] this->bins;
		this->bins = NULL;
		this->cnt = 0;
	}

	void clear(int init=0) {
		memset(this->bins,init,sizeof(int)*this->cnt);
	}

	unsigned max() {
		int max = 0;

		for(int i=0; i < this->cnt; i++) {
			if (this->bins[max] < this->bins[i])
				max = i;
		}

		return max;
	}

	unsigned min() {
		int min = 0;

		for(int i=0; i < this->cnt; i++) {
			if (this->bins[min] > this->bins[i])
				min = i;
		}

		return min;
	}

	string toString() {
		string str = "Histogram(";
		char buffer[10];

		sprintf(buffer,"%d",this->cnt);
		str += buffer;
		str += ") = {";

		for(int i=0; i < this->cnt; i++) {
			sprintf(buffer,"%d",this->bins[i]);
			str += buffer;
			if (i != this->cnt-1)
				str += ",";
		}

		return str + "}";
	}

	int& operator[] (const unsigned i) const {
		if (i >= this->cnt)
			throw "Subscript out of bounds!";

		return this->bins[i];
	}

	Histogram& operator=(const Histogram &rhs) {
		if (this == &rhs)
			return *this;

		delete [] this->bins;

		this->cnt = rhs.cnt;
		this->bins = new int[this->cnt];
		memcpy(this->bins,rhs.bins,sizeof(int) * this->cnt);

		return *this;
	}

	Histogram& operator+=(const Histogram &rhs) {
		if (this->cnt != rhs.cnt)
			throw "Histogram sizes don't match!";

		for(int i=0; i < this->cnt; i++)
			this->bins[i] += rhs.bins[i];

		return *this;
	}

	Histogram& operator-=(const Histogram &rhs) {
		if (this->cnt != rhs.cnt)
			throw "Histogram sizes don't match!";

		for(int i=0; i < this->cnt; i++)
			this->bins[i] -= rhs.bins[i];

		return *this;
	}

	const Histogram operator+(const Histogram &rhs) {
		return Histogram(*this) += rhs;
	}

	const Histogram operator-(const Histogram &rhs) {
		return Histogram(*this) -= rhs;
	}

private:

	int* bins;
	int  cnt;
};

#endif
