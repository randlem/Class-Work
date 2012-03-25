/**********************************************************
*
* CS529 Lab 2 - Frame-based 3 way comms
*
* Mark Randles
* 2009-07-30
* 
* PURPOSE: To study an implement a frame-based comms system
* between four processes in a star network.
*
* ARCHITECTURE: There's a single parent process and three
* child processes which are spawned from the parent process
* Each child sends messages until they're done then they just
* respond with ACKs.  The parent recieves messages then 
* responds with messages until it's done then it just routes
* recieved messages.
*
**********************************************************/

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

/**********************************************************
* PROGRAM DEFINES
**********************************************************/
#define USAGE_MESSAGE 	"Usage: lab2 [n] [m1] [m2] [m3]"
#define CHILD			1
#define PARENT			0
#define FRAME_TYPE_DATA 1
#define FRAME_TYPE_ACK  0

/**********************************************************
* MACROS
**********************************************************/
#define RANDRANGE(a,b)	((rand()%(b-a+1))+a)
#define SWAP(a,b)		((a) ^= (b) ^= (a) ^= (b))

/**********************************************************
* TYPEDEFS
**********************************************************/
typedef unsigned char uchar;

// packet data structure
typedef union {
	struct {
		uchar source_addr;
		uchar dest_addr;
		uchar frame_type;
		int number;
	} parts;
	uchar buffer[8];
} packet_t;

/**********************************************************
* GLOBAL VALUES
**********************************************************/
static int		n = 0;
static int		m[] = {0,0,0,0};
static pid_t	pids[] = {0,0,0,0};

/**********************************************************
* FUNCTION DECLARES
**********************************************************/
bool doChild(int, int[2]);
bool doParent(int[4][2]);
void printEvenOdd(int, int);

int main(int argc, char *argv[]) {
	pid_t pid;
	int id,ret,sockets[4][2];

	// clear the socket data structure
	memset(sockets,0,sizeof(int)*4*2);

	// make sure the cmd line is well formed
	if (argc != 5) {
		cerr << USAGE_MESSAGE << endl;
		exit(-1);
	}

	// get the cmd line args
	n = atoi(argv[1]);
	m[1] = atoi(argv[2]);
	m[2] = atoi(argv[3]);
	m[3] = atoi(argv[4]);

	// check to make sure the values are sane
	if (n <= 0) {
		cerr << "The entered n was less then or equal to zero." << endl;
		exit(-1);
	}
	
	if (m[1] <= 0 || m[2] <= 0 || m[3] <= 0) {
		cerr << "The entered m was less then or equal to zero." << endl;
		exit(-1);
	}
	
	// set some intial data values
	pids[0] = getpid();
	id		= 0;
	
	// fork the process three times
	for(int i=1; i <= 3; i++) {
		// generate the socketpair
		if ((ret = socketpair(AF_UNIX,SOCK_STREAM,0,sockets[i])) == -1) {
			perror("socketpair");
			exit(-1);
		}

		// fork
		if ((pid = fork()) == -1) {
			perror("fork");
			exit(-1);
		}
		
		// if we're the child do the child function
		if (pid == 0) {
			doChild(i,sockets[i]);	
			exit(0);
		}
		
		// otherwise log the pid
		pids[i] = pid;
		cout << "Parent created child process with pid " << pid << endl;
	}

	// do the parent thang
	doParent(sockets);

	exit(0);
}

void printEvenOdd(int id, int number) {
	cout << "P" << id << ": The number " << number << " is "
		 << (((number % 2) == 0) ? "even" : "odd") << "." << endl;
}

void printPacketCreation(packet_t &p) {
	cout << "P" << (int)p.parts.source_addr << ": Is sending " << p.parts.number << " to P" 
		 << (int)p.parts.dest_addr << "." << endl;
}

bool doChild(int id, int socket[2]) {
	int sent = m[id];
	packet_t packet;
	
	// seed the random number gen and close the parents socket
	srand(time(NULL) + id);
	close(socket[PARENT]);

	// loop until I've sent all my messages
	while (sent != 0) {
		// create a new packet to a random destination
		packet.parts.source_addr = id;
		packet.parts.dest_addr = RANDRANGE(0,3);
		packet.parts.frame_type = FRAME_TYPE_DATA;
		packet.parts.number = RANDRANGE(1,1000);

		printPacketCreation(packet);
		
		// write the new packet and wait for the response
		write(socket[CHILD],&packet.buffer,sizeof(packet_t));
		sent--;
		read(socket[CHILD],&packet.buffer,sizeof(packet_t));

		// based on the recieved packet type, do something
		switch (packet.parts.frame_type) {
			case FRAME_TYPE_DATA: {
				printEvenOdd(id,packet.parts.number);
			} break;
			case FRAME_TYPE_ACK:
			default: { } break;
		}
	}

	cout << "P" << id << ": Is done" << endl;
	
	// pack the write stream with an ACK packet to singal the child is done
	packet.parts.source_addr = id;
	packet.parts.dest_addr = 0;
	packet.parts.number = 0;
	packet.parts.frame_type = FRAME_TYPE_ACK;
	write(socket[CHILD],&packet.buffer, sizeof(packet_t));
	
	// loop and process new packets until we get the stop packet
	while (true) {
		read(socket[CHILD],&packet.buffer,sizeof(packet_t));

		// respond to the stop packet with a stop packet
		if (packet.parts.number == -1) {
			SWAP(packet.parts.source_addr,packet.parts.dest_addr);
			packet.parts.number = -1;
			write(socket[CHILD],&packet.buffer,sizeof(packet_t));
			break;
		}

		SWAP(packet.parts.source_addr,packet.parts.dest_addr);
		packet.parts.number = 0;
		packet.parts.frame_type = FRAME_TYPE_ACK;
		write(socket[CHILD],&packet.buffer, sizeof(packet_t));
	}

	// shutdown and exit cleanly
	close(socket[CHILD]);
	exit(0);
}

bool doParent(int sockets[4][2]) {
	int i,child,random,sent=n;
	packet_t packet;
	bool process_done[4] = {false,false,false,false}, route = false, finished = false;

	// seed the random generator and close the child sockets
	srand(time(NULL));
	for(i=1; i < 4; i++) {
		close(sockets[i][CHILD]);
	}

	// loop until everybody is done
	while (!finished) {
		// get an unfinished child id
		while (process_done[child = RANDRANGE(1,3)]);
				
		// read the packet they've written
		read(sockets[child][PARENT],&packet.buffer,sizeof(packet_t));
		
		// based on the read packet type do something
		switch (packet.parts.frame_type) {
			case FRAME_TYPE_DATA: {
				// if this packet is for me, process it and respond with a packet or ACK
				if (packet.parts.dest_addr == 0) {
					printEvenOdd(0,packet.parts.number);
					
					SWAP(packet.parts.source_addr, packet.parts.dest_addr);
					if (sent > 0) {
						packet.parts.frame_type = FRAME_TYPE_DATA;
						packet.parts.number = RANDRANGE(1,1000);
						sent--;
						printPacketCreation(packet);
					} else {
						packet.parts.frame_type = FRAME_TYPE_ACK;
					}
					
					write(sockets[packet.parts.dest_addr][PARENT],&packet.buffer,sizeof(packet_t));
				} else {
					// pass the packet on to the recipient and respond to the originator with a
					// packet or an ACK
					packet.parts.source_addr = 0;
					write(sockets[packet.parts.dest_addr][PARENT],&packet.buffer,sizeof(packet_t));
					
					packet.parts.dest_addr = child;
					if (sent > 0) {
						packet.parts.frame_type = FRAME_TYPE_DATA;
						packet.parts.number = RANDRANGE(1,1000);
						sent--;
						printPacketCreation(packet);
					} else {
						packet.parts.frame_type = FRAME_TYPE_ACK;
					}
					write(sockets[packet.parts.dest_addr][PARENT],&packet.buffer,sizeof(packet_t));
				}
			} break;
			case FRAME_TYPE_ACK: {
				// if we get an ACK from the child, mark them done
				process_done[child] = true;
			} break;
		}
		
		// see if the children are done
		finished = true;
		for (i=1; i < 4; i++)
			finished &= process_done[i];
	}
	
	while (sent > 0) {
		packet.parts.source_addr = 0;
		packet.parts.dest_addr = RANDRANGE(1,3);
		packet.parts.frame_type = FRAME_TYPE_DATA;
		packet.parts.number = RANDRANGE(1,1000);
		
		printPacketCreation(packet);
		write(sockets[i][PARENT],&packet.buffer,sizeof(packet_t));
		sent--;
		read(sockets[i][PARENT],&packet.buffer,sizeof(packet_t));
	}
	
	cout << "P0: Sent all messages." << endl;
	
	// kill all the threads
	packet.parts.number = -1;
	for (i=1; i < 4; i++) {
		packet.parts.dest_addr = i;
		write(sockets[i][PARENT],&packet.buffer,sizeof(packet_t));
		read(sockets[i][PARENT],&packet.buffer,sizeof(packet_t));
		close(sockets[i][PARENT]);
	}
	
	exit(0);
}

