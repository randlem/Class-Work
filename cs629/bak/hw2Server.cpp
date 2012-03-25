#include <iostream>
using std::cout;
using std::endl;
using std::ios;

#include <map>
using std::map;

#include <vector>
using std::vector;

#include <string>
using std::string;

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <errno.h>
#include <string.h>

#define READ_PIPE	0
#define WRITE_PIPE	1

const string USAGE_MESSAGE = "USAGE: hw2Server <listening port>";
const string DEFAULT_LISTEN_PORT = "1337";
const int MAX_RECV_SIZE = 65536;

struct ServerThread {
	int 				socket;
	sockaddr_storage	socket_addr;
	int 				pipe[2];
	pthread_t			thread;
};

map<string, ServerThread*>	active_servers;

//map<string, int>				active_sockets;
//map<string, sockaddr_storage*>	active_sockets_addr;
//map<string, int*>				active_sockets_pipe;
//map<string, pthread_t>			threads;

void *serverWorker(void *);

inline string trimStr(const string& src, const string& c = " \r\n")
{
	int p2 = src.find_last_not_of(c);
	if (p2 == std::string::npos)
			return std::string();
	int p1 = src.find_first_not_of(c);
	if (p1 == std::string::npos)
			p1 = 0;
	return src.substr (p1, (p2 - p1) + 1);
}

void * getInAddr(const struct sockaddr *sa) {
	if (sa->sa_family == AF_INET)
		return &(((struct sockaddr_in *)sa)->sin_addr);

	return &(((struct sockaddr_in6 *)sa)->sin6_addr);
}

int getInPort(const struct sockaddr *sa) {
	if (sa->sa_family == AF_INET)
		return (((struct sockaddr_in *)sa)->sin_port);

	return (((struct sockaddr_in6 *)sa)->sin6_port);
}

int explode(vector<string>* parts, char *str, const char* delims) {
	char *part = NULL;

	parts->clear();
	part = strtok(str,delims);
	while (part != NULL) {
		parts->push_back(string(part));
		part = strtok(NULL,delims);
	}

	return parts->size();
}

int main(int argc, char* argv[]) {
	string listenPort = "0";
	int ret;
	int sock = 0;
	struct addrinfo hints;
	struct addrinfo *addr;
	int next_id = 1;
	char buffer[80];
	int* temp_pipe = NULL;
	ServerThread* new_thread = NULL;

	// check to see if too many arguments were passed
	if (argc > 2) {
		cout << USAGE_MESSAGE << endl;
		return -1;
	}

	// if one arg is present, that means that no args were passed
	if (argc == 1) {
		listenPort = DEFAULT_LISTEN_PORT;
	} else { // otherwise something was specified
		// convert the passed port
		listenPort = argv[1];
		int lP = atoi(listenPort.c_str());

		if (lP < 1024 && lP > 0) {
			cout << "You want me to try and bind on port " << listenPort
				 << " this is a restricted port, so I'm going to try, but it's"
				 << "going to fail." << endl;
		} else if (lP > 65535 || lP <= 0) {
			cout << "Port number out of range.  Try something between 1024 and"
				 << " 65535.  You can go lower then that if you're brave."
				 << endl;
		}
	}

	// setup the getaddrinfo() hints struct, and auto-make my addrinfo struct
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;	// use whatever IP proto that's default
	hints.ai_socktype = SOCK_STREAM;	// we want a stream socket
	hints.ai_flags = AI_PASSIVE;	// give me my IP

	ret = getaddrinfo(NULL, listenPort.c_str(), &hints, &addr);
	if (0 != ret) {
		cout << "getaddrinfo: " << gai_strerror(ret) << endl;
		return -2;
	};

	// create the socket
	sock = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
	if (-1 == sock) {
		perror("socket");
		return -2;
	}

	// bind to the created socket
	ret = bind(sock, addr->ai_addr, addr->ai_addrlen);
	if (-1 == ret) {
		perror("bind");
		return -2;
	}

	cout << "Socket created on " << listenPort << endl;

	// cleanup some unneeded memory
	freeaddrinfo(addr);

	// wait till I get a connection then spawn a new pthread
	while (true) {
		ret = listen(sock,5);
		if (ret != 0) {
			perror("listen");
			return -2;
		}

		// create the next client id
		memset(buffer,0,80);
		sprintf(buffer,"c%d",next_id);
		string client_id = buffer;
		next_id++;

		// create a new ServerThread object
		new_thread = new ServerThread;

		// accept the connection & get the new socket
		socklen_t connected_addr_len = sizeof(new_thread->socket_addr);
		int conn_fd = accept(sock, (struct sockaddr*)&new_thread->socket_addr,
			&connected_addr_len);
		if (conn_fd == 0) {
			perror("accept");
			return -2;
		}

		// resolve the client address
		memset(buffer,0,80);
		inet_ntop(new_thread->socket_addr.ss_family,
            getInAddr((struct sockaddr *)&new_thread->socket_addr),
            buffer, 80);

		cout << "Recieved and accepted a client request from " << buffer
			 << " on port "
			 << getInPort((struct sockaddr *)&new_thread->socket_addr) << endl;

		// store the connected addr and connection fd
		new_thread->socket = conn_fd;

		// create the pipe for this thread
		ret = pipe(new_thread->pipe);
		if(-1 == ret) {
			perror("pipe");
			return -2;
		}

		// save the ServerThread structure
		active_servers[client_id] = new_thread;

		// spawn off the pthread and pass the socket
		ret = pthread_create(&new_thread->thread, NULL,
			serverWorker, (void*) &client_id);
	}

	close(sock);

	return 0;
}

void *serverWorker(void *c_id) {
	string client_id = ((string*)c_id)->c_str();
	string send_to = "";
	ServerThread* this_thread = active_servers[client_id];
	ServerThread* other_thread = NULL;
	int ret = -1;
	char buffer[MAX_RECV_SIZE];
	char c;
	string msg;
	string timestamp;
	FILE *in_pipe = NULL;
	FILE *out_pipe = NULL;

	cout << "Child thread " << client_id << " is running." << endl;

	// recieve who the messages are going to
	memset(buffer,0,MAX_RECV_SIZE);
	ret = recv(this_thread->socket,buffer,MAX_RECV_SIZE,0);
	switch (ret) {
		case -1: {
			perror("recv");
			pthread_exit(NULL);
		} break;
		case 0: {
			cout << "Server thread " << client_id
				 << " recived a hang-up." << endl;
		} break;
		default: {
			vector<string> parts;
			explode(&parts,buffer,"|");

			send_to = parts[0];
			timestamp = parts[1];
			msg = parts[2];
			msg = timestamp + " " + msg;
		} break;
	}
	cout << "Server for " << client_id
		 << " sending messages to " << send_to << endl;

	// block until the other thread comes up
	while(active_servers[send_to] == NULL) { ; }

	// setup some variables
	other_thread = active_servers[send_to];
	in_pipe = fdopen(this_thread->pipe[READ_PIPE], "r");
	out_pipe = fdopen(other_thread->pipe[WRITE_PIPE], "w");

	// write the message to the pipe
	fputs(msg.c_str(),out_pipe);
	fclose(out_pipe);

	// read anything waiting in my pipe
	string send_msg = "";
	while ((c = fgetc (in_pipe)) != EOF)
		send_msg += c;

	// send message to client
	send(this_thread->socket,send_msg.c_str(),send_msg.length(),0);

	// clean up pipe file descriptors & socket
	fclose(in_pipe);
	close(this_thread->socket);

	cout << "Child thread " << client_id << " is exiting." << endl;
	pthread_exit(NULL);
}
