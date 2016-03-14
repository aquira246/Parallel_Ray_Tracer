#pragma  once
#ifndef __Ray__
#define __Ray__

#include <Eigen/Dense>

class Ray
{
	public:
		Ray();
		Ray(Eigen::Vector3f d);
		Ray(Eigen::Vector3f e, Eigen::Vector3f d);
		~Ray();

		Eigen::Vector3f eye, direction;

	private:
};

#endif
