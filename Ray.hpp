#pragma  once
#ifndef __RAY_H__
#define __RAY_H__

#include <cuda_runtime.h>
#include "Vector3f.h"
#include "VectorMath.h"

class Ray
{
	public:
      __device__ __host__
      Ray();
      __device__ __host__
      Ray(Vector3f d);
      __device__ __host__
      Ray(Vector3f e, Vector3f d);
      __device__ __host__
      ~Ray();

		Vector3f eye, direction;

	private:
};

#endif
