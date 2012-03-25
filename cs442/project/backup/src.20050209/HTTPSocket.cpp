#include "HTTPSocket.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdlib.h>
#include <unistd.h>
#include <strings.h>

HTTPSocket::HTTPSocket(unsigned short size) : bufferSize(size)
{
	buffer=new char[bufferSize];
	isConnected=false; isEOF=false;
	handle=-1;
}

HTTPSocket::~HTTPSocket()
{
	if(isConnected)
		::close(handle);
	delete[] buffer;
}

bool HTTPSocket::open(string remoteHost, unsigned short remotePort)
{
	struct sockaddr_in target;
	struct hostent* hostIP;

	hostIP=gethostbyname(remoteHost.c_str());
	if(hostIP==NULL)
		return false;

	memcpy((char *)&target.sin_addr,hostIP->h_addr,hostIP->h_length);
	target.sin_family=AF_INET;
	target.sin_port=htons(remotePort);

	if((handle=socket(AF_INET,SOCK_STREAM,0))<0)
		return false;

	if(connect(handle,(struct sockaddr *)&target,sizeof(target))<0)
		return close();

	isConnected=true;

	return true;
}

bool HTTPSocket::close()
{
	isConnected=false;
	::close(handle);
	return false;
}

HTTPSocket& HTTPSocket::operator<<(string message)
{
	send(handle,message.c_str(),bufferSize*sizeof(char),0);

	return *this;
}

HTTPSocket& HTTPSocket::operator>>(string& destination)
{
	unsigned int bytesRead=0;

	do
	{
		bzero(buffer,bufferSize*sizeof(char));
		bytesRead=recv(handle,buffer,bufferSize*sizeof(char),0);
		destination+=buffer;
	} while(bytesRead>0);

	return *this;
}
