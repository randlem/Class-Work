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
using std::ios;

#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/socket.h>

/**********************************************************
* PROGRAM DEFINES
**********************************************************/
#define USAGE_MESSAGE 	"Usage: lab3 [n] [m]"

#define CHILD			1
#define PARENT			0

#define FRAME_TYPE_DATA 1
#define FRAME_TYPE_ACK  0
#define FRAME_TYPE_NACK 2

#define FCS_POLY		0xE3

#define STATS_SENT		0
#define STATS_RECV		1
#define STATS_DATA_SENT 1
#define STATS_DATA_ACK  0
#define STATS_DATA_NACK 2

/**********************************************************
* MACROS
**********************************************************/
#define RANDRANGE(a,b)	((rand()%(b-a+1))+a)
#define SWAP(a,b)		((a) ^= (b) ^= (a) ^= (b))

// CRC defines
#define WIDTH			(8 * sizeof(fcs_t))
#define TOPBIT			(1 << (WIDTH - 1))

/**********************************************************
* TYPEDEFS
**********************************************************/
typedef unsigned char			uchar;
typedef short int				sint;
typedef uchar					fcs_t;

// packet data structure
typedef union {
	struct {
		sint 	number;
		uchar	source_addr;
		uchar	dest_addr;
		uchar	frame_type;
		fcs_t	fcs;
	} parts;
	uchar data[6];
} packet_t;

/**********************************************************
* GLOBAL VALUES
**********************************************************/
int stats[2][2][3];
int n, m;
int comms[2];

/**********************************************************
* FUNCTION DECLARES
**********************************************************/
void print_stats(int);
void do_child();
void do_parent();
fcs_t compute_fcs(packet_t&);
bool check_fcs(packet_t&);
void send_packet(packet_t&);
void recv_packet(packet_t&);
void print_send_msg(packet_t&);
void print_recv_msg(packet_t&);

/**********************************************************
* FUNCTION IMPLEMENTATIONS
**********************************************************/
int main (int argc, char* argv[]) {
	int child_pid, ret;
	
	// process the command line
	if (argc != 3) {
		cout << USAGE_MESSAGE << endl;
		exit(-1);
	}
	
	n = atoi(argv[1]);
	m = atoi(argv[2]);
	
	if (n <= 0 || m <= 0) {
		cout << "Both n and m must be larger then 0." << endl;
		exit(-2);
	}
		
	// clear the stats memory
	memset(stats,0,sizeof(int)*2*2*3);
	
	// create the socketpair
	if ((ret = socketpair(AF_UNIX,SOCK_STREAM,0,comms)) == -1) {
		perror("socketpair");
		exit(-1);
	}
	
	// fork and run the processes
	switch (fork()) {
		case -1: {
			perror("fork");
			exit(-1);
		} break;
		case 0: {
			do_child();
		} break;
		default: {
			do_parent();
		} break;
	}
	
	exit(0);	
}

void print_stats(int id) {
	cout << "Stats for Process " << id << ":" << endl
		 << "\tSENT:" << endl
		 << "\t\tData Packets: " << stats[id][STATS_SENT][STATS_DATA_SENT] << endl
		 << "\t\tACK Packets: " << stats[id][STATS_SENT][STATS_DATA_ACK] << endl
		 << "\t\tNACK Packets: " << stats[id][STATS_SENT][STATS_DATA_NACK] << endl
		 << "\tRECV:" << endl
		 << "\t\tData Packets: " << stats[id][STATS_RECV][STATS_DATA_SENT] << endl
		 << "\t\tACK Packets: " << stats[id][STATS_RECV][STATS_DATA_ACK] << endl
		 << "\t\tNACK Packets: " << stats[id][STATS_RECV][STATS_DATA_NACK] << endl;
}

// modified from algo 1 at: 
// http://www.netrino.com/Embedded-Systems/How-To/CRC-Calculation-C-Code
fcs_t compute_fcs(packet_t& p) {
	fcs_t remainder = 0;
	int byte,bit;
	
	for (byte=0; byte < sizeof(packet_t)-sizeof(fcs_t); ++byte) {
		remainder ^= (p.data[byte] << (WIDTH - 8));
		
		for (bit = 8; bit > 0; --bit) {
			if (remainder & TOPBIT)
				remainder = (remainder << 1) ^ FCS_POLY;
			else
				remainder = (remainder << 1);
		}
	}
	
	return remainder;
}

// evolved from computFCS(), see for cite
bool check_fcs(packet_t& p) {
	fcs_t remainder;
	int byte,bit;
	
	for (byte=0; byte < sizeof(packet_t); byte++) {
		remainder ^= (p.data[byte] << (WIDTH - 8));
		
		for (bit = 8; bit > 0; bit--) {
			if (remainder & TOPBIT)
				remainder = (remainder << 1) ^ FCS_POLY;
			else
				remainder = (remainder << 1);
		}
	}
	
	return (remainder == 0);
}

void print_recv_msg(packet_t& p) {
	cout << "P" << (int)p.parts.dest_addr << ": The number " 
		 << p.parts.number << " is "
		 << (((p.parts.number % 2) == 0) ? "even" : "odd") 
		 << "." << endl;
}

void print_send_msg(packet_t& p) {
	cout << "P" << (int)p.parts.source_addr << ": Is sending " 
		 << p.parts.number << " to P" 
		 << (int)p.parts.dest_addr << "." << endl;
}

void send_packet(packet_t& p) {
	p.parts.fcs = compute_fcs(p);
	
	// invalidate one in 1000 packets
	if (RANDRANGE(0,9) == 0)
		p.parts.fcs = p.parts.fcs << RANDRANGE(1,7);
	
	write(comms[p.parts.source_addr],&p.data,sizeof(packet_t));
	stats[p.parts.source_addr][STATS_SENT][p.parts.frame_type]++;
}

void recv_packet(uchar id, packet_t *p) {
	packet_t spacket;
	
	memset(p->data,0,sizeof(packet_t));
	
	while(true) {
		read(comms[id],&p->data,sizeof(packet_t));
		stats[id][STATS_RECV][p->parts.frame_type]++;
		
		if (check_fcs(*p))
			break;
		
		cout << "P" << (int)id << ": Recvd an invalid packet sending back NACK!" << endl;
		
		spacket.parts.dest_addr		= p->parts.source_addr;
		spacket.parts.source_addr	= p->parts.dest_addr;
		spacket.parts.frame_type	= FRAME_TYPE_NACK;
		send_packet(spacket);
	}
}

void do_child() {
	packet_t spacket,rpacket;
	int sent = m;
	bool done = false;
	
	srand(getpid());
	
	spacket.parts.source_addr = CHILD;
	spacket.parts.dest_addr = PARENT;
	
	while (!done) {
		recv_packet(CHILD,&rpacket);
		
		switch (rpacket.parts.frame_type) {
			case FRAME_TYPE_DATA: {
				if (rpacket.parts.number == -1) {
					cout << "P1: Recvd die frame, exiting!" << endl;
					done = true;
					spacket.parts.frame_type = FRAME_TYPE_DATA;
					spacket.parts.number = -1;
					print_stats(CHILD);
					break;
				} else {
					print_recv_msg(rpacket);
				}
			}
			case FRAME_TYPE_ACK: {
				if (sent > 0) {
					spacket.parts.frame_type = FRAME_TYPE_DATA;
					spacket.parts.number = RANDRANGE(1,1000);
					sent--;
					print_send_msg(spacket);
				} else {
					spacket.parts.frame_type = FRAME_TYPE_ACK;
					spacket.parts.number = 0;
				}
			} break;
			case FRAME_TYPE_NACK: {
				// retransmit last packet
			} break;
			default: break;
		}
		
		send_packet(spacket);
	}
	
	exit(0);
}

void do_parent() {
	packet_t spacket,rpacket;
	int sent = n;
	bool child_done = false, retransmit = false;

	srand(getpid());
	
	// generate the first packet to send
	spacket.parts.source_addr = PARENT;
	spacket.parts.dest_addr = CHILD;
	
	while (sent > 0 || !child_done || retransmit) {
		if (!retransmit) {
			if (sent > 0) {
				spacket.parts.frame_type = FRAME_TYPE_DATA;
				spacket.parts.number = RANDRANGE(1,1000);
				sent--;
				print_send_msg(spacket);
			} else {
				spacket.parts.frame_type = FRAME_TYPE_ACK;
				spacket.parts.number = 0;
			}
		}
		
		send_packet(spacket);
		recv_packet(PARENT,&rpacket);
		retransmit = false;
		
		switch (rpacket.parts.frame_type) {
			case FRAME_TYPE_NACK: {
				retransmit = true;
			} break;
			case FRAME_TYPE_ACK: {
				child_done = true;
				cout << "P0: Child done!" << endl;
			} break;
			case FRAME_TYPE_DATA: {
				print_recv_msg(rpacket);
			} break;
			default: break;
		}
	}
	
	spacket.parts.frame_type = FRAME_TYPE_DATA;
	spacket.parts.number = -1;
	
	send_packet(spacket);
	recv_packet(PARENT,&rpacket);
	
	print_stats(PARENT);
	exit(0);
}
