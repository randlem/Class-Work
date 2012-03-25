################################################################################
#
# CS525 -- Spline Manipulation with Tcl/Tk
#
# Written By: Mark Randles
#
# Description:
#     This program is designed to gather 4 points from the user and draw the
# curve defined by them using Cubic Hermite form.  The points p0 and p1 are
# start and end points.  The points p0t and p1t are tangent points to p0 and p1.
#
# Archtecture:
#     The primary architecture of this program is button actions.  The primary
# function of the program is accessed by clicking the "Draw" button on the main
# window.  When this button is clicked a new window is opened up that allow the
# user to input the 4 points required to draw the curve.  This window has two
# buttons, one will cancel the draw action and the other will pass the data
# values to the drawer.
#     At any time the user can exit the program by clicking the "Quit" button
# or by selecting "Quit" from the menu.
#
################################################################################

# setup the main window
wm title . "CS525 -- Program 5 -- Spline Manipulation"
wm resizable . 0 0

# create a menu frame and the draw canvas
frame .f_top -height 0.5i -width 6i -bg LemonChiffon
canvas .c_bom -height 5.5i -bg white

# create the menu button
menubutton .f_top.mb -text File -fg MediumBlue -bg LightSkyBlue -menu .f_top.mb.menu1
set m1 [menu .f_top.mb.menu1 -tearoff 0]
$m1 add command -label Quit -command exit

# create the draw button and bind it to the draw proc
button .c_bom_draw -text "\nDraw\n" -fg SaddleBrown -bg orange -activebackground LightGoldenrod
bind .c_bom_draw <Button-1> {draw_spline}

# create the quit button and bind it to the exit func
button .c_bom_quit -text "\nExit\n" -fg DarkSalmon -bg DarkRed -activebackground red
bind .c_bom_quit <Button-1> {exit}

# place and pack the menubar
place .f_top.mb -x 0.1i -y 0.1i
pack .f_top -side top

# place and pack the draw button
place .c_bom_draw -x 0.8i -y 5.3i
pack .c_bom -fill x

# place and pack the quit button
place .c_bom_quit -x 0.1i -y 5.3i
pack .c_bom -fill x

# globals
set ::dlgCoords 0
set ::p0x 0
set ::p0y 0
set ::p1x 0
set ::p1y 0
set ::p0tx 0
set ::p0ty 0
set ::p1tx 0
set ::p1ty 0

# proc to draw the spline
proc draw_spline {} {
	# get the coords from the user
	get_coords
	tkwait variable dlgCoords
	if {$::dlgCoords == 2} {
		return
	}
	destroy .coords

	# draw the axis on the screen
	draw_axis

	# draw the hermite spline
	draw_hermite
}

# proc to draw the axis
proc draw_axis {} {
	.c_bom create line 0.2i 2.6i 5.8i 2.6i -fill green
	.c_bom create line 3.0i 0.2i 3.0i 5.3i -fill green

	for {set j 0.2} {$j <= 5.8} {set j [expr $j + 0.2]} {
		.c_bom create line ${j}i 2.65i ${j}i 2.55i -fill green
	}

	for {set j 0.2} {$j <= 5.3} {set j [expr $j + 0.2]} {
		.c_bom create line 2.95i ${j}i 3.05i ${j}i -fill green
	}
}

proc draw_hermite {} {
	set last_x $::p0x
	set last_y $::p0y

	# plot the points as 5x5 rects
	.c_bom create rectangle [expr $::p0x-2] [expr $::p0y-2] [expr $::p0x+2] [expr $::p0y+2] -fill blue
	.c_bom create rectangle [expr $::p1x-2] [expr $::p1y-2] [expr $::p1x+2] [expr $::p1y+2] -fill blue
	.c_bom create rectangle [expr $::p0tx-2] [expr $::p0ty-2] [expr $::p0tx+2] [expr $::p0ty+2] -fill blue
	.c_bom create rectangle [expr $::p1tx-2] [expr $::p1ty-2] [expr $::p1tx+2] [expr $::p1ty+2] -fill blue

	# loop through a normalized t and draw the line
	for {set t 0} {$t <= 1} {set t [expr $t + 0.01 ] } {
		# precompute some values of t
		set t3 [expr $t*$t*$t]
		set t2 [expr $t*$t]

		# compute the hermite functions
		set h00 [expr (2*$t3) - (3*$t2) + 1]
		set h10 [expr $t3 - (2*$t2) + $t]
		set h01 [expr (-2*$t3) + (3*$t2)]
		set h11 [expr $t3 - $t2]

		# compute the x and y coords for the new line point
		set x [expr ($h00*$::p0x) + ($h10*$::p1x) + ($h01*$::p0tx) + ($h11*$::p1tx)]
		set y [expr ($h00*$::p0y) + ($h10*$::p1y) + ($h01*$::p0ty) + ($h11*$::p1ty)]

		# draw the line
		.c_bom create line $last_x $last_y $x $y -fill red

		# store the points for the next line draw
		set last_x $x
		set last_y $y
	}
}

proc get_coords {} {
	# create a new window
	toplevel .coords
	wm title .coords "Hermite Points"
	wm resizable . 0 0

	# create a frame
	frame .coords.c

	# create widgets for p0
	label .coords.c.p0x_lbl -text "Point #1 X"
	entry .coords.c.p0x -width 3
	label .coords.c.p0y_lbl -text "Point #1 Y"
	entry .coords.c.p0y -width 3

	# create widgets for p1
	label .coords.c.p1x_lbl -text "Point #2 X"
	entry .coords.c.p1x -width 3
	label .coords.c.p1y_lbl -text "Point #2 Y"
	entry .coords.c.p1y -width 3

	# create widgets for p0 tangent
	label .coords.c.p0tx_lbl -text "Tangent #1 X"
	entry .coords.c.p0tx -width 3
	label .coords.c.p0ty_lbl -text "Tangent #1 Y"
	entry .coords.c.p0ty -width 3

	# create widgets for p1 tangent
	label .coords.c.p1tx_lbl -text "Tangent #2 X"
	entry .coords.c.p1tx -width 3
	label .coords.c.p1ty_lbl -text "Tangent #2 Y"
	entry .coords.c.p1ty -width 3

	# create an "Ok" & "Cancel" button and bind them to their functions
	button .coords.c.ok -text Okay
	bind .coords.c.ok <Button-1> {onOk}
	button .coords.c.cancel -text Cancel
	bind .coords.c.cancel <Button-1> {onCancel}

	# bind window destroy to cancel function
	bind .coords <Destroy> {onCancel}

	# position the frame
	grid .coords.c -column 0 -row 0

	# position widget for p0
	grid .coords.c.p0x_lbl -column 1 -row 0
	grid .coords.c.p0x -column 2 -row 0
	grid .coords.c.p0y_lbl -column 3 -row 0
	grid .coords.c.p0y -column 4 -row 0

	# position widget for p1
	grid .coords.c.p1x_lbl -column 1 -row 1
	grid .coords.c.p1x -column 2 -row 1
	grid .coords.c.p1y_lbl -column 3 -row 1
	grid .coords.c.p1y -column 4 -row 1

	# position widget for p0 tangent
	grid .coords.c.p0tx_lbl -column 1 -row 2
	grid .coords.c.p0tx -column 2 -row 2
	grid .coords.c.p0ty_lbl -column 3 -row 2
	grid .coords.c.p0ty -column 4 -row 2

	# position widget for p1 tangent
	grid .coords.c.p1tx_lbl -column 1 -row 3
	grid .coords.c.p1tx -column 2 -row 3
	grid .coords.c.p1ty_lbl -column 3 -row 3
	grid .coords.c.p1ty -column 4 -row 3

	# position buttons
	grid .coords.c.ok -column 1 -row 4 -columnspan 2
	grid .coords.c.cancel -column 3 -row 4 -columnspan 2

	# proc if "Ok" button is pressed
	proc onOk {} {
		set ::dlgCoords 1

		# set the globals with the entered values
		set ::p0x [.coords.c.p0x get]
		set ::p0y [.coords.c.p0y get]
		set ::p1x [.coords.c.p1x get]
		set ::p1y [.coords.c.p1y get]
		set ::p0tx [.coords.c.p0tx get]
		set ::p0ty [.coords.c.p0ty get]
		set ::p1tx [.coords.c.p1tx get]
		set ::p1ty [.coords.c.p1ty get]
	}

	# proc if "Cancel" button is pressed
	proc onCancel {} {
		set ::dlgCoords 2
	}
}
