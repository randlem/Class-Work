/*
*** FILE NAME   : cs_456_250_setup.h
*** DESCRIPTION : This is a header file to be used in CS4250/5250/6250 assignments.
                  It contains initialization function calls and setups for
				  OpenGL programs with GLUT.  The initializations involve
				  a callback hander definition which sets viewing paraleters.
*** DATE        : Jan. 2009
*** WRITTEN By  : JKL
*/

//@@***********************************************************************************@@
void reshape_handler(int width, int height)
{
	float ratio = 1.0 * width / ((height == 0) ? 1 : height);
	glViewport(0, 0, width, height);						// sets the viewport

	glMatrixMode(GL_PROJECTION);							// projection matrix
	glLoadIdentity();										// loads identity matrix
	gluPerspective(45, ratio, 0.0, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

}	// end of reshape_handler()

//@@***********************************************************************************@@
void init_setup(int width, int height, char *windowName)
{
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA); // double buffer, rgba color
	glutInitWindowSize(width, height);			  // init. window size
	glutInitWindowPosition(256, 256);			  // init. window position
	glutCreateWindow(windowName);				  // window name
	glutReshapeFunc(reshape_handler);		      // sets the reshape call back

	// sets blending up for alpha channel transparency
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// setup the depth buffer
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	// setup the lighting
//	glEnable(GL_LIGHTING);
//	glEnable(GL_LIGHT0);
//	glShadeModel(GL_SMOOTH);

}	// end of init_setup()
