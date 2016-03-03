#pragma  once
#ifndef __Box__
#define __Box__

#include <Eigen/Dense>
#include <vector>
#include "Shape.hpp"

typedef struct hit_struct {
   Shape *hitShape;
   double t;
   bool isHit;
} hit_t;

class Box
{
	public:
		Box();
		Box(Eigen::Vector3f c);
		Box(float w, float h, float d);
		Box(Eigen::Vector3f c, float w, float h, float d);
		~Box();
		
		Eigen::Vector3f center;
		Eigen::Vector3f basePt;
		float width, height, depth; // x, y, z

		hit_t checkHit(Ray testRay);

		std::vector<Shape *> shapes;
		void addShape(Shape *s);


	private:
};

#endif
