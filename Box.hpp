#pragma  once
#ifndef __Box__
#define __Box__

#include <Eigen/Dense>
#include <Sphere.hpp>
#include <vector>

typedef struct hit_struct {
   Sphere *hitSphere;
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

		std::vector<Sphere> spheres;
		void addSphere(Sphere s);
		void addSphere(Eigen::Vector3f center, float radius);


	private:
};

#endif
