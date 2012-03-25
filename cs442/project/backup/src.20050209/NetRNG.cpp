#include <iostream>
using std::cerr;
using std::endl;

#include <ctype.h>
#include "NetRNG.h"

pthread_t NetRNG::myThread;
pthread_mutex_t NetRNG::mutex;
unsigned short NetRNG::activeRNG=0;
bool NetRNG::ready=false;
HTTPSocket NetRNG::uplink;
BoundedBuffer<double> NetRNG::numberBuffer(20);

NetRNG::NetRNG()
{
	if(!NetRNG::activeRNG++)
		pthread_create(&NetRNG::myThread,NULL,NetRNG::start,NULL);
}

NetRNG::~NetRNG()
{
	if(!--NetRNG::activeRNG)
		NetRNG::stop();
}

void * NetRNG::start(void * nothing)
{
	NetRNG::ready=true;

	while(fetchSet()){}

	return NULL;
}

void NetRNG::stop()
{
	numberBuffer.tearDown();
	pthread_join(NetRNG::myThread,NULL);
	NetRNG::ready=false;
}

bool NetRNG::fetchSet()
{
	string resultSet;
	unsigned short ct=0;
	unsigned int tempVal;

	uplink.open("www.random.org");

	if(!uplink)
		return false;

	uplink << "GET /cgi-bin/checkbuf HTTP/1.0\n\n";
	uplink >> resultSet;

	// you need to check the buffer on the remote site to see if there are numbers
	// to get to avoid hanging the request.  They recommend to avoid this if less then 20%
	resultSet.erase(resultSet.size() - 2);
	resultSet.erase(0,resultSet.size() - 2);
	while(atoi(resultSet.c_str()) < 20) { cerr << resultSet << endl; }

	uplink << "GET /cgi-bin/randnum?num=100&min=0&max=100000&col=1 HTTP/1.0\n\n";
	uplink >> resultSet;

	char * myHandle = const_cast<char *>(resultSet.c_str())+resultSet.find("\r\n\r\n")+4;

	while(ct<100)
	{
		tempVal=0;

		while(!isdigit(*myHandle)){myHandle++;}

		while(isdigit(*myHandle)){
			tempVal*=10;
			tempVal+=*myHandle-'0';
			myHandle++;
		}


		if(!numberBuffer.put(((double)tempVal)/100000))
			return false;

		ct++;
	}

	uplink.close();
	return true;
}
