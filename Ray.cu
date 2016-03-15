#include "Ray.hpp"
#include <cuda_runtime.h>

__device__ __host__
Ray::Ray()
{
	direction = Vector3f(0,0,1);
	eye = Vector3f(0,0,0);
}

__device__ __host__
Ray::Ray(Vector3f d)
{
	direction = d;
	eye = Vector3f(0,0,0);
}

__device__ __host__
Ray::Ray(Vector3f e, Vector3f d)
{
	direction = d;
	eye = e;
}

__device__ __host__
Ray::~Ray()
{
	
}
