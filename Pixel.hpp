#pragma  once
#ifndef __Pixel__
#define __Pixel__

#include <cuda_runtime.h>

class Pixel
{
	public:
      __device__ __host__
		Pixel();
      __device__ __host__
		Pixel(float iR, float iG, float iB);
      __device__ __host__
		~Pixel();

		float r, g, b;

      __device__ __host__
		void Average(float newR, float newG, float newB);
      __device__ __host__
		void AveragePx(Pixel other);
      __device__ __host__
		bool HasColor();

	private:
};

#endif
