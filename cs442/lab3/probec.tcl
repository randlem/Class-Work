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

# create 6 nodes
#Node set multiPath_ 1
set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]
set n5 [$ns node]
set n6 [$ns node]

$n0 set multiPath_ 1
$n1 set multiPath_ 1
$n2 set multiPath_ 1
$n3 set multiPath_ 1
$n4 set multiPath_ 1
$n5 set multiPath_ 1
$n6 set multiPath_ 1

# create the network
$ns simplex-link $n0 $n1 10Mb 1000ms DropTail
$ns simplex-link $n1 $n2 10Mb 100ms DropTail
$ns simplex-link $n1 $n3 10Mb 100ms DropTail
$ns simplex-link $n2 $n4 10Mb 100ms DropTail
$ns simplex-link $n2 $n5 10Mb 100ms DropTail
$ns simplex-link $n3 $n4 10Mb 100ms DropTail
$ns simplex-link $n3 $n5 10Mb 100ms DropTail
$ns simplex-link $n4 $n6 10Mb 100ms DropTail
$ns simplex-link $n5 $n6 10Mb 100ms DropTail

$ns simplex-link-op $n0 $n1 orient right
$ns simplex-link-op $n1 $n2 orient right-up
$ns simplex-link-op $n1 $n3 orient right-down
$ns simplex-link-op $n2 $n4 orient right
$ns simplex-link-op $n2 $n5 orient right-down
$ns simplex-link-op $n3 $n4 orient right-up
$ns simplex-link-op $n3 $n5 orient right
$ns simplex-link-op $n4 $n6 orient right-down
$ns simplex-link-op $n5 $n6 orient right-up

$ns rtproto DV

set udp0 [new Agent/UDP]
$udp0 set class_ 1
$ns attach-agent $n0 $udp0

set exp0 [new Application/Traffic/CBR]
$exp0 set packetSize_ 5000
$exp0 set interval_ 0.005
$exp0 attach-agent $udp0

set null6 [new Agent/Null]
$ns attach-agent $n6 $null6

$ns connect $udp0 $null6

$ns at 0.001 "$exp0 start"
$ns at 6.000 "finish"

$ns run