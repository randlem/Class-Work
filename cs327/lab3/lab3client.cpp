/***********************************************
 * lab3client.cpp - client for finger protocol
 *
 * Written by Mark Randles
 *
 * PURPOSE: To explore the wide world of network programming
 *  by writing a finger client.  Will query a finger server
 *  with the approiate query.  I will follow Dr. Zimmerman's cmd line
 *  input specification.  You must supply a domain and port to make
 *  this program work.  Usernames are optional.
 *
 * USAGE: lab3client domain port [username1] [username2] [username3] ...
 *            l   WHOIS mode.  "Long mode" for other finger clients.
 *
 ***********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream.h>
#include <string.h>
#include <memory.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define CRLF 0x0D0A

struct sockaddr_in dest_addr;
char** user_list;
int numb_users;

int init(int argv, char* argc[]);
int connect();

int main(int argv, char* argc[]) {
	int sockfd;            // socket file descriptor
	char buffer[4096];
    
    // do the initialization routine
	if(init(argv,argc)) {
        printf("Invalid cmd line usage. Make sure you included a hostname and port\n");
        return(1);
    }

    if(numb_users == 0) {
        char request[2] = { 0x0D, 0x0A };
        memset(buffer,0,4096);        

        // connect to the remote
        if((sockfd = connect()) == -1) {
            return(1);
        }
        
        // send the request
        send(sockfd,request,2,0);
        
        // wait for the responce
        recv(sockfd,buffer,4096,0);
        
        // output the response
        printf("%s\n",buffer);
        
        close(sockfd);
        
    } else {
        for(int i=0; i < numb_users; i++) {
            char* request;
            memset(buffer,0,4096);
            
            // connect to the remote host
            if((sockfd = connect()) == -1) {
                return(1);
            }
            
            // build the request
            request = new char[strlen(user_list[i])+2];
            strcpy(request,user_list[i]);
            request[strlen(user_list[i])] = 0x0D;
            request[strlen(user_list[i])+1] = 0x0A;
         
            // send the request
            send(sockfd,request,strlen(user_list[i])+2,0);
        
            // wait for the responce
            if(0 == recv(sockfd,buffer,4096,0)) {
                printf("Remote host closed socket.\n");
                break;
            }
        
            // output the response
            printf("%s\n",buffer);
            
            close(sockfd);
            
            delete[] request;
        }
    }
            
    // deallocate the memory
    for(int i=0; i < numb_users; i++)
        delete[] user_list[i];
    delete[] user_list;
    
	return(0);
}

int init(int argv, char* argc[]) {
	if(argv < 3) return(1);
	
	struct hostent* host;
    
    host = gethostbyname(argc[1]);
    
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(atol(argc[2]));
    dest_addr.sin_addr.s_addr = inet_addr(inet_ntoa(*((struct in_addr *)host->h_addr)));
    memset(&(dest_addr.sin_zero), '\0', 8); 
	
    if(argv > 3) {
        user_list = new (char*)[argv - 3];
        for(int i=3; i < argv; i++) {
            user_list[i-3] = new char[strlen(argc[i])+1];
            strcpy(user_list[i-3],argc[i]);
        }
        numb_users = argv-3;
    }
    
	return(0);
}

int connect() {
    int sockfd;

    // open a BSD style socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd == -1) {
        printf("Couldn't get a socket.\n");
        return(-1);
    }

    // don't forget your error checking for bind():
    if(-1 == connect(sockfd, (struct sockaddr *)&dest_addr, sizeof(struct sockaddr))) {
        printf("Couldn't connect to remote socket.\n");
        return(-1);
    }
    
    return(sockfd);
}