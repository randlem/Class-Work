#include "cqueue.h"

CalendarQueue::CalendarQueue(double w) : CalendarQueue(DEFAULT_COUNT, w) {

}

CalendarQueue::CalendarQueue(int N, double w) {
	bucketCount = resizeThreshold = N;
	bucketWidth = w;
	elementCount = 0;
}

CalendarQueue::~CalendarQueue() {

}

bool CalendarQueue::push(double p) {

}

bool CalendarQueue::pop() {

}

bool CalendarQueue::peak() {

}

void CalendarQueue::resize() {

}
