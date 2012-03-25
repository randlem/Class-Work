
//@@***********************************************************************************@@
void reshape_handler(int width, int height)
{
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);
}	// end of reshape_handler()


//@@***********************************************************************************@@
void init_setup(int width, int height, const char *windowName)
{
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(5, 5);
	glutCreateWindow(windowName);
	glutReshapeFunc(reshape_handler);
}	// end of init_setup()
