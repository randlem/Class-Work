/*
*** FILE NAME   : cs_425_525_setup.h
*** DESCRIPTION : This is a header file to be used in CS425/525 assignments.
                  It contains initialization function calls and setups for
				  OpenGL programs with GLUT.  The initializations involve
				  a callback hander definition which sets viewing paraleters.
*** DATE        : Jan. 2009
*** WRITTEN By  : JKL
*/

//@@***********************************************************************************@@
void reshape_handler(int width, int height)
{
	glViewport(0, 0, width, height);							// sets the viewport
	glMatrixMode(GL_PROJECTION);								// projection matrix
	glLoadIdentity();											// loads identity matrix
	gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);	// 2D orthographic projection
}	// end of reshape_handler()

//@@***********************************************************************************@@
void init_setup(int width, int height, char *windowName)
{
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);  // single buffer, rgba color
	glutInitWindowSize(width, height);			  // init. window size
	glutInitWindowPosition(5, 5);				  // init. window position
	glutCreateWindow(windowName);				  // window name
	glutReshapeFunc(reshape_handler);		      // sets the reshape call back
	glEnable(GL_BLEND); 						  // sets blending up for alpha channel transparency
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}	// end of init_setup()
