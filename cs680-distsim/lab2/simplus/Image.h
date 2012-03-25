#include <string>
using std::string;

#include "png.h"
#include "stdio.h"
#include "math.h"

#ifndef IMAGE_H
#define IMAGE_H

typedef struct {
	int x;
	int y;
} point;

typedef struct {
	unsigned char r;
	unsigned char g;
	unsigned char b;
} rgb;

enum ImageType {PNG};

class Image {
	public:
		Image(int,int,string = "image");
		~Image();

		void clearImage(rgb&);
		void setFileName(string);

		void drawPoint(point&, rgb&);
		void drawLine(point&, point&, rgb&);
		void drawLine(point&, point&, int, rgb&);
		void drawRect(point&, point&, rgb&);

		bool outputImage(ImageType = PNG);

	private:
		bool outputPNG(string);

		int width;
		int height;
		string fileName;

		rgb** raster;
};

#endif
