#ifndef CQUEUE_H
#define CQUEUE_H

const int DEFAULT_COUNT = 16;

class CalendarQueue {
	public:
		CalendarQueue(double w);
		CalendarQueue(int N, double w);
		~CalendarQueue();

		bool push(double p);
		bool pop();
		bool peak();

	private:
		void resize();

		int bucketCount;
		int resizeThreshold;
		double bucketWidth;
		int elementCount;
};

#endif