#pragma  once
#ifndef __SPHERE_H__
#define __SPHERE_H__

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include "Shape.hpp"
#include "Ray.hpp"
#include "VectorMath.h"
#include "Vector3f.h"

class Sphere: public Shape
{
	public:
		Sphere();
		Sphere(Vector3f c);
		Sphere(float r);
		Sphere(Vector3f c, float r);
		~Sphere();
		
		// Shape has a center and radius, the only components of a sphere

      virtual Vector3f GetNormal(Vector3f hitPt);
		float checkHit(Vector3f eye, Vector3f dir);

	private:
};

#endif
