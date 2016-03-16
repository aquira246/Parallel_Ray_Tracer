#pragma  once
#ifndef __Picture__
#define __Picture__

#include <Pixel.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include "Image.hpp"

class Picture
{
	public:
		Picture();
		Picture(int w, int h);
		~Picture();
		
		int width, height;
		std::vector<Pixel> pixels;

		void setPixel(int x, int y, Pixel newP);
		Pixel getPixel(int x, int y);
		void resize(int w, int h);

		void Print(std::string fileName);


	private:
		int getIdx(int x, int y);
};

#endif
