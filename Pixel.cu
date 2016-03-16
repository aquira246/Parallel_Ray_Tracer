#include <Pixel.hpp>

__device__ __host__
Pixel::Pixel()
{
	r = 0;
	g = 0;
	b = 0;
}

__device__ __host__
Pixel::Pixel(float iR, float iG, float iB)
{
	r = iR;
	g = iG;
	b = iB;
}

__device__ __host__
Pixel::~Pixel()
{
	
}

__device__ __host__
void Pixel::Average(float newR, float newG, float newB)
{
	r += newR;
	r = r/2.0f;

	g += newB;
	g = g/2.0f;

	b += newB;
	b = b/2.0f;
}

__device__ __host__
void Pixel::AveragePx(Pixel other)
{
	r += other.r;
	r = r/2.0f;

	g += other.g;
	g = g/2.0f;

	b += other.b;
	b = b/2.0f;
}

__device__ __host__
bool Pixel::HasColor() {
	if (r <= .0001 && g <= .0001 && b <= .0001) {
		return false;
	}

	return true;
}
