#pragma  once
#ifndef __Pixel__
#define __Pixel__

#include <Eigen/Dense>

class Pixel
{
	public:
		Pixel();
		Pixel(float iR, float iG, float iB);
		~Pixel();
		float r, g, b;
		void Average(float newR, float newG, float newB);
		void AveragePx(Pixel other);
		bool HasColor() ;

	private:
};

#endif
