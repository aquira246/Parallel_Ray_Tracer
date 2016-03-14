#pragma  once
#ifndef __Ray__
#define __Ray__

#include "Vector3f.h"
#include "VectorMath.h"

class Ray
{
	public:
		Ray();
		Ray(Vector3f d);
		Ray(Vector3f e, Vector3f d);
		~Ray();

		Vector3f eye, direction;

	private:
};

#endif
