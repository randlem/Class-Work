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

class Fractional {
	friend ostream& operator<<(ostream& output, const Fractional& f) {
//		output << setw(8);

		if (f.denominator == 1 || f.numerator == 0)
			output << f.numerator;
		else if (f.denominator == 0)
			output << "NaN" << endl;
		else {
			if (Fractional::output_float) {
//				output << fixed << setw(8) << setfill(' ') << setprecision(2);
				output << (double)((double)f.numerator/(double)f.denominator);
//				output.unsetf(ios::fixed);
			} else
				output << f.numerator << "/" << f.denominator;
		}

		return output;
	}

	public:
		static bool output_float;

		Fractional(int num, int den) : numerator(num), denominator(den) { }
		Fractional(int num) : numerator(num), denominator(1) { }
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
			int num = (numerator * f.denominator) -	 (f.numerator * denominator);

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

		int getInt() const {
			return numerator;
		}

	protected:
		int numerator;
		int denominator;
};

#endif
