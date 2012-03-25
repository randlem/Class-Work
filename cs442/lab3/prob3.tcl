# create a simulator object
set ns [new Simulator]

#define different colors for the data flows
$ns color 1 Blue
$ns color 2 Red

# open the name trace file
set nf [open out.nam w]
$ns namtrace-all $nf

#open the trace file
set tr [open out.tr w]
$ns trace-all $tr

#define a finish proc
proc finish {} {
	global ns nf tr
	$ns flush-trace

	close $nf
	close $tr

	exec nam out.nam &

	exit 0
}

# create six nodes
set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]
set n5 [$ns node]

# create links between the nodes
$ns duplex-link $n0 $n1 1.5Mb 10ms DropTail
$ns duplex-link $n1 $n2 1.5Mb 10ms DropTail
$ns duplex-link $n2 $n3 1.5Mb 10ms DropTail
$ns duplex-link $n1 $n4 1.5Mb 10ms DropTail
$ns duplex-link $n3 $n4 1.5Mb 10ms DropTail
$ns duplex-link $n4 $n5 0.5Mb 10ms DropTail

$ns duplex-link-op $n0 $n1 orient right
$ns duplex-link-op $n1 $n2 orient up
$ns duplex-link-op $n2 $n3 orient right
$ns duplex-link-op $n1 $n4 orient right
$ns duplex-link-op $n3 $n4 orient down
$ns duplex-link-op $n4 $n5 orient right

# create a TCP agent and attach it to n0
set tcp0 [new Agent/TCP]
$tcp0 set class_ 1 # set the class color
$ns attach-agent $n0 $tcp0

# create a FTP source
set ftp0 [new Application/FTP]
$ftp0 attach-agent $tcp0

# create at UDP agent and attach it to n3
set udp3 [new Agent/UDP]
$udp3 set class_ 2
$ns attach-agent $n3 $udp3

# create CBR source at n3
set cbr3 [new Application/Traffic/CBR]
$cbr3 set packetSize_ 500
$cbr3 set interval_ 0.005
$cbr3 attach-agent $udp3

# create a null agent (sink) at n5
set null5 [new Agent/Null]
$ns attach-agent $n5 $null5

# create a TCPSink agent at n5
set tcpsink5 [new Agent/TCPSink]
$ns attach-agent $n5 $tcpsink5

# attach the sources to the sink
$ns connect $tcp0 $tcpsink5
$ns connect $udp3 $null5

# schedule initial events
$ns at 0.05 "$ftp0 start"
$ns at 1.00 "$ftp0 stop"
$ns at 0.2  "$cbr3 start"
$ns at 0.8  "$cbr3 stop"
$ns at 1.1  "finish"

# bring down the link from n3 to n4 at 0.5
$ns rtmodel-at 0.5 down $n3 $n4

# set the routing model
$ns rtproto Session

$ns run
