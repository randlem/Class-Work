#include "Image.h"

int main() {
	Image image(640,480);
	point p0 = { 320, 240 };
	point p1 = { 600, 400 };
	rgb white = { 255, 255, 255 };
	rgb black = { 0, 0, 0 };
	rgb red = { 255, 0, 0 };
	rgb blue = { 0, 0, 255 };
	rgb green = { 0, 255, 0 };

	image.clearImage(white);
/*
	image.drawPoint(p0,red);

	p0.x = 321;
	p0.y = 241;
	image.drawPoint(p0,blue);
*/
	p0.x = 100;
	p0.y = 100;
	p1.x = 200;
	p1.y = 200;
	image.drawLine(p0,p1,black);

	p0.x = 200;
	p0.y = 100;
	p1.x = 100;
	p1.y = 200;
	image.drawLine(p0,p1,green);

	p0.x = 200;
	p0.y = 200;
	p1.x = 100;
	p1.y = 100;
	image.drawLine(p0,p1,red);

	p0.x = 100;
	p0.y = 200;
	p1.x = 200;
	p1.y = 100;
	image.drawLine(p0,p1,blue);

/*	p0.x = 100;
	p0.y = 100;
	p1.x = 150;
	p1.y = 150;
	image.drawRect(p0,p1,green);
*/
	image.outputImage();

	return(0);
}
