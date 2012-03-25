#ifndef __FRACTIONAL_H__
#define __FRACTIONAL_H__

#include <ostream>
using std::ostream;
using std::ios;

#include <iomanip>
using std::setprecision;
using std::setw;
using std::setfill;
using std::fixed;

#include <string>
using std::string;

class Fractional {
	friend ostream& operator<<(ostream& output, const Fractional& f) {
		if (f.denominator == 1 || f.numerator == 0)
			output << f.numerator;
		else if (f.denominator == 0)
			output << "NaN" << endl;
		else
			output << f.numerator << "/" << f.denominator;

		return output;
	}

	public:
		Fractional(int num, int den) : numerator(num), denominator(den) { }
		Fractional(int num) : numerator(num), denominator(1) { }
		Fractional(const Fractional &f) {
			numerator = f.numerator;
			denominator = f.denominator;
		}
		Fractional(string f) {
			int pos = f.find("/");

			if (pos == string::npos) {
				numerator = atoi(f.c_str());
				denominator = 1;
			} else {
				numerator = atoi(f.substr(0,pos).c_str());
				denominator = atoi(f.substr(pos+1).c_str());
			}

			simplify();
		}
		~Fractional() { }

		static int gcd(int a, int b) {
			if (b == 0)
				return a;

			return gcd(b, a % b);
		}

		Fractional simplify() {
			if (numerator == 0) {
				denominator = 1;
				return *this;
			}

			reduce();

			if (denominator < 0) {
				denominator *= -1;
				numerator *= -1;
			}

			return *this;
		}

		Fractional reduce() {
			double gcd = Fractional::gcd(numerator,denominator);

			numerator /= gcd;
			denominator /= gcd;

			return *this;
		}

		Fractional inverse() {
			int t = denominator;
			denominator = numerator;
			numerator = t;

			return *this;
		}

		Fractional operator+ (Fractional f) const {
			if (f.denominator == denominator)
				return Fractional(numerator + f.numerator, denominator);

			int den = f.denominator * denominator;
			int num = (numerator * f.denominator) -	(f.numerator * denominator);

			return Fractional(num,den).simplify();
		}

		Fractional operator- (Fractional f) const {
			if (f.denominator == denominator)
				return Fractional(numerator - f.numerator, denominator).simplify();

			int den = f.denominator * denominator;
			int num = (numerator * f.denominator) - (f.numerator * denominator);

			return Fractional(num,den).simplify();
		}

		Fractional operator* (Fractional f) const {
			return Fractional(numerator * f.numerator, denominator * f.denominator).simplify();
		}

		Fractional operator/ (Fractional f) const {
			f.inverse();

			return Fractional(numerator * f.numerator, denominator * f.denominator).simplify();
		}

		void operator= (const Fractional f)  {
			numerator = f.numerator;
			denominator = f.denominator;
		}

		Fractional operator+ (int i) const {
			return Fractional(numerator + i * denominator, denominator).simplify();
		}

		Fractional operator- (int i) const {
			return Fractional(numerator - i * denominator, denominator).simplify();
		}

		Fractional operator* (int i) const {
			return Fractional(numerator * i, denominator).simplify();
		}

		Fractional operator/ (int i) const {
			return Fractional(numerator, denominator * i).simplify();
		}

		bool operator> (const Fractional f) const {
			return (numerator * f.denominator) > (denominator * f.numerator);
		}

		bool operator< (const Fractional f) const {
			return (numerator * f.denominator) < (denominator * f.numerator);
		}

		bool operator== (const Fractional f) const {
			return (numerator == f.numerator) && (denominator == f.denominator);
		}

		int getInt() const {
			return numerator;
		}

		double getFloat() const {
			return (double)((double)numerator / denominator);
		}

	protected:
		int numerator;
		int denominator;
};

#endif
