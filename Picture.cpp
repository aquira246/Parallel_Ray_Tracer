#include <Picture.hpp>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

Picture::Picture()
{
	width = 0;
	height = 0;
	Pixels.clear();
}

Picture::Picture(int w, int h)
{
	width = 0;
	height = 0;
	Pixels.clear();

	resize(w, h);
}

Picture::~Picture()
{
	Pixels.clear();
}

void Picture::setPixel(int x, int y, Pixel newP)
{
	int idx = getIdx(x, y);
	if (idx != -1)
		Pixels[idx] = newP;
}

Pixel Picture::getPixel(int x, int y)
{
	int idx = getIdx(x, y);
	if (idx != -1)
		return Pixels[idx];
	else {
		cout << "ERROR! Index out of bounds" << endl;
		return Pixels[0];
	}
}

void Picture::resize(int w, int h)
{
	int oldWidth = width;
	int oldHeight = height;
	width = w;
	height = h;
	Pixels.resize(width*height);

	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < height; ++y)
		{
			if (x >= oldWidth && y >= oldHeight) {
				Pixels[getIdx(x, y)] = Pixel();
			}
		}
	}
}

int Picture::getIdx(int x, int y) {
	int ret = 0;
	ret = x*width + y;
	return ret;
}

void Picture::Print(string fileName) {
	//ofstream myfile;
	//myfile.open(fileName.c_str());
	
	Image *img;
	color_t clr;
	Pixel temp;

	img = new Image(width, height);
	
	for (int y = height - 1; y >= 0; --y)
	 {
		for (int x = 0; x < width; ++x)	
		{
			temp = getPixel(x, y);
			clr.r = temp.r;
			clr.g = temp.g;
			clr.b = temp.b;
			clr.f = 1.0;
			img->pixel(x, y, clr);

			/*if (getPixel(x, y).HasColor()) {
				myfile << "*";
			} else {
				myfile << " ";
			}*/
		}
		//myfile << "\n";
	}

	//myfile.close();
	char *holdName = (char *)fileName.c_str();
	img->WriteTga(holdName, true); 
}
