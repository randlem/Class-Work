#include <iostream>
#include "SimPlus.h"

using namespace std;

int main()
{
	SimPlus* handle = SimPlus::getInstance();
	for(;;)
		handle->scheduleEvent( handle->getEvent() );
	delete handle;
	return 0;
}
