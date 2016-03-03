#pragma  once
#ifndef __Sphere__
#define __Sphere__

#include <Shape.hpp>
#include <Eigen/Dense>
#include <math.h>
#include <Ray.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 

class Sphere: public Shape
{
	public:
		Sphere();
		Sphere(Eigen::Vector3f c);
		Sphere(float r);
		Sphere(Eigen::Vector3f c, float r);
		~Sphere();
		
		// Shape has a center and radius

		float checkHit(Ray ray) {
            return checkHit(ray.eye, ray.direction);
      }

		float checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir);


	private:
};

#endif
