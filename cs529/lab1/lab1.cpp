#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/socket.h>

// program defines
#define USAGE_MESSAGE 	"Usage: lab1 [n] [m]"
#define CHILD			1
#define PARENT			0

// macros
#define RANDRANGE(a,b)	((rand()%(b-a+1))+a)

// data structures

typedef union {
	struct {
		short int number;
		unsigned short int address;
	} parts;
	unsigned int packed;
	unsigned char buffer[4];
} packet_t;

// shared global variables
static int		n = 0;
static int		m = 0;
static pid_t	pids[] = {0,0,0,0};

bool doChild(int, int[2]);
bool doParent(int[4][2]);

int main(int argc, char *argv[]) {
	pid_t pid;
	int id,ret,sockets[4][2];

	memset(sockets,0,sizeof(int)*4*2);

	if (argc != 3) {
		cerr << USAGE_MESSAGE << endl;
		exit(-1);
	}

	n = atoi(argv[1]);
	m = atoi(argv[2]);

	srand(time(NULL));

	if (n <= 0) {
		cerr << "The entered n was less then or equal to zero." << endl;
		exit(-1);
	}
	
	if (m <= 0) {
		cerr << "The entered m was less then or equal to zero." << endl;
		exit(-1);
	}
	
	pids[0] = getpid();
	id		= 0;
	for(int i=1; i <= 3; i++) {
		if ((ret = socketpair(AF_UNIX,SOCK_STREAM,0,sockets[i])) == -1) {
			perror("socketpair");
			exit(-1);
		}

		if ((pid = fork()) == -1) {
			perror("fork");
			exit(-1);
		}
		
		if (pid == 0) {
			doChild(i,sockets[i]);	
			exit(0);
		}
		
		pids[i] = pid;
		cout << "Parent created child process with pid " << pid << endl;
	}

	doParent(sockets);

	exit(0);
}

bool doChild(int id, int socket[2]) {
	int sent = m;
	packet_t packet;
	
	srand(time(NULL));
	close(socket[PARENT]);
	
	while (true) {
		read(socket[CHILD],&packet.buffer,4);

		if (packet.parts.number < 0)
			break;
		
		cout << "P" << id << ": The number " << packet.parts.number << " is " 
			 << (((packet.parts.number % 2) == 0) ? "even" : "odd") << "." << endl;

		if (sent > 0) {
			packet.parts.number = RANDRANGE(1,1000);
			while ((packet.parts.address = RANDRANGE(0,3)) == id);

			cout << "P" << id << ": The number " << packet.parts.number 
				 << " being passed is for P" << packet.parts.address << endl;
			
			write(socket[CHILD],&packet.buffer,4);
			sent--;
		} else {
			packet.parts.number = 0;
			packet.parts.address = 0;
			write(socket[CHILD],&packet.buffer,4);			
		}
	}

	packet.parts.number = -1;
	packet.parts.address = 0;
	write(socket[CHILD],&packet.buffer,4);
	
	close(socket[CHILD]);
	exit(0);
}

bool doParent(int sockets[4][2]) {
	int i,child,random,sent=n;
	packet_t packet;
	bool child_done[4] = {false,false,false,false}, route = false, finished = false;

	srand(time(NULL));
	for(i=1; i < 4; i++) {
		close(sockets[i][CHILD]);
	}

	while (sent > 0) {
		if (route) {
			write(sockets[packet.parts.address][PARENT],&packet.buffer,4);
		} else {
			child = RANDRANGE(1,3);
			random = RANDRANGE(1,1000);

			cout << "P0: The number " << random 
				 << " being passed is for P" << child << endl;

			packet.parts.number = random;
			packet.parts.address = child;

			write(sockets[child][PARENT],&packet.buffer,4);
			sent--;
		}

		read(sockets[packet.parts.address][PARENT],&packet.buffer,4);
		
		// if child returns an ACK packet, we're going to kill them
		// since they should have exhausted their sent values
		// we mark them true in the child_done array to skip them
		// during packet generation
		if (packet.parts.number == 0) {
			child_done[child] = true;
			route = false;
			cout << "P0: Recieved ACK packet from P" << child << endl;
		} else {
			if (packet.parts.address == 0) {
				cout << "P0: The number " << packet.parts.number << " is " 
					 << (((packet.parts.number % 2) == 0) ? "even" : "odd") << "." << endl;
				route = false;
			} else {
				cout << "Routing packet from P" << child << " to P" << packet.parts.address << endl;
				route = true;
			}
		}
	}
	
	cout << "P0: Sent all messages." << endl;
	
	// kill all the threads
	packet.parts.number = -1;
	for (i=1; i < 4; i++) {
		packet.parts.address = i;
		write(sockets[i][PARENT],&packet.buffer,4);
		read(sockets[i][PARENT],&packet.buffer,4);
		close(sockets[i][PARENT]);
	}
	
	exit(0);
}

