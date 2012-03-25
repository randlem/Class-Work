#include <iostream>
#include <GL/glut.h>
#include "WinTimer.h"
#include "Timer.h"
#include "Image.h"
#include "PPMReader.h"
#include "cs_456_250_setup.h"
using namespace std;

const int WINDOW_WIDTH = 500;
const int WINDOW_HEIGHT = 500;
const char* WINDOW_TITLE = "Skeleton";

Image img;
WinTimer wt;
Timer t;

void display_func()
{
    glClear(GL_COLOR_BUFFER_BIT);

    //glDrawPixels(img.width, img.height, GL_RGB, GL_UNSIGNED_BYTE, img.buf);
    
    glBegin(GL_POINTS);
        for (int y = 0; y < img.height; y++)
        {
            for (int x = 0; x < img.width; x++)
            {
                unsigned char* rgb = img.GetPixelAt(x, y);
                glColor3ub(rgb[0], rgb[1], rgb[2]);
                glVertex2i(x, y);
            }
        }
    glEnd();
    
    
    glFlush();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'q':
        exit(0);
    }
}

//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
	// *** Start of algorithm code ***
    t.Start();

	// Declarations
	char file[256];
    char fileIn[128], fileOut[128];
	int currentFrame[64] = {0};
	int previousFrame[64] = {0};
	int differenceArray [6784];
	int primaryThreshold = 0; 
	int secondaryThreshold = 0;
	long sum = 0;
	
	// We know there are 6783 frames to read with consistent naming,
	// iterate through each frame and populate the histogram array for
	// frame i, then compare it to frame i-1 and save the difference
	for (int index = 1; index <= 6783; index++)
	{
		// We first need to populate the image object by reading in
		// a file name based on the loop counter
		sprintf(file, "converted/project-video%.4db.ppm", index);

		// If there is a file read error then stop execution
		if (!ReadPPM(img, file))
		{
			printf("Something went wrong!\n");
			return 0;
		}

		// Find the color code for each pixel and update the histogram
		// for this frame based on the bin that the color code falls in
        for (int y = 0; y < img.height; y++)
        {
            for (int x = 0; x < img.width; x++)
            {
                unsigned char* rgb = img.GetPixelAt(x, y);
                int bin = ((int)rgb[0] / 12) + ((int)rgb[1] / 12) + ((int)rgb[2] / 12);
				currentFrame[bin]++;
            }
        }

		int difference = 0;
		if (index > 1)
		{
			for (int j = 0; j < 64; j++)
			{
				difference += abs(previousFrame[j] - currentFrame[j]);
			}

			// Add the total difference between the two frames to the difference
			// array - this is all of the information we'll need from the input
			// files going forward
			differenceArray[index-1] = difference;
		}

		// We also need a running total to compute the average at the end of
		// the loop - this will be used for threshold calculation
		sum += difference;

		// Finally we need to get currentFrame and previousFrame ready for
		// the next iteration
		for (int j = 0; j < 64; j++)
		{
			previousFrame[j] = currentFrame[j];
			currentFrame[j] = 0;
		}
		
	}
	
	// Compute the average sum difference between frames and find
	// threshold values based on this average
	float averageDifference = sum / 6783.0;
	cout << "average difference: " << averageDifference << endl;

	// *** need to calculate the thresholds here ***

	// Start looking for transitions
	
	// *** more to come ***

    t.Stop();
	// *** End of algorithm code ***

    cout << "Elapsed time: " << t.GetElapsedSeconds() << endl;

	glutInit(&argc, argv);
    init_setup(img.width, img.height, WINDOW_TITLE);

    glutDisplayFunc(display_func);
    glutKeyboardFunc(keyboard);

    glutMainLoop();
}