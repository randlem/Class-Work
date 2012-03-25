// HTTPSocket.h
//
// A C++ class that sends and receives STL strings
// via TCP/IP--designed as an HTTP client but will
// probably work for other TCP-based protocols.
// Uses a ton of C include files to provide networking
// functionality, and may require linking 
// with -lxnet on the compile line.
// 
// METHODS:
// --------
//
// HTTPSocket(unsigned short)
// Constructor; argument is size of char[] buffer used to read
// from the socket--probably no effect, so it defaults to 1024
// and tinkering is not recommended.
//
// ~HTTPSocket()
// Destructor; deallocates buffer & closes socket if it's open.
//
// open(string, unsigned short) : bool
// Opens a connection to the specified server, at the specified
// port.  The port defaults to 80 since this is HTTPSocket.
// Server may be a FQDN or an IP address.
//
// close() : bool
// Closes the open socket; returns false for convenience, not
// to indicate an error condition.
//
// operator!() : bool
// Returns false if the socket connection is open, true if closed.
//
// eof() : bool
// Returns true if socket is in EOF state.  Provided for compatibility
// with iostream interface, but not currently in use.
//
// operator<<(string) : HTTPSocket&
// Overloaded stream insertion operator; used to send message across
// socket to server.  Should be cascadable.
//
// operator>>(string) : HTTPSocket&
// Overloaded stream extraction operator; used to get a message
// from the server over TCP.

#include <cstdlib>
#include <netdb.h>
#include <unistd.h>
#include <sys/socket.h>
#include <strings.h>
#include <string>
using std::string;

#ifndef HTTPSOCKET
#define HTTPSOCKET

class HTTPSocket{

	public:

		HTTPSocket(unsigned short size=1024);
		~HTTPSocket();

		bool open(string remoteHost, unsigned short remotePort=80);
		bool close();

		inline bool operator!(){	return !isConnected; }
		inline bool eof(){		return isEOF; }

		HTTPSocket& operator<<(string message);
		HTTPSocket& operator>>(string& dest);

	private:
		bool isConnected,isEOF;
		char * buffer;
		int handle;
		const unsigned short bufferSize;
};

#endif
