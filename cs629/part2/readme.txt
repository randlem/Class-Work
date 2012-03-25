===================================
SERVER
===================================
randlem@vpr ~ $ ./a.out
Socket created on 1337
Recieved and accepted a client request from ::ffff:98.31.63.143 on port 31464
c1: is running.
c1: sending messages to c2
Recieved and accepted a client request from ::ffff:98.31.63.143 on port 31976
c2: is running.
c2: sending messages to c1
c2: 20081101-19:09:02:0817 asdf
c2: finished sending message to client.
c1: 20081101-19:09:06:0939 qwer
c1: finished sending message to client.
c1: recieved ack from client. Exiting
c1: is done.
c2: recieved ack from client. Exiting
c2: is done.

===================================
Apache Log
===================================
98.31.63.143 - - [01/Nov/2008:19:09:52 -0400] "GET /cs629/remoteClass.class HTTP/1.1" 200 2183
98.31.63.143 - - [01/Nov/2008:19:09:56 -0400] "GET /cs629/remoteClass.class HTTP/1.1" 200 2183

===================================
Client #1
===================================
randlem@boole ~/class/compsci/cs629/part2 $ java hw2Client2 c2 asdf
20081101-19:09:06:0939 qwer

===================================
Client #2
===================================
randlem@boole ~/class/compsci/cs629/part2 $ java hw2Client2 c1 qwer
20081101-19:09:02:0817 asdf
