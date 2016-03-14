#include <Ray.hpp>

Ray::Ray()
{
	direction = Eigen::Vector3f(0,0,1);
	eye = Eigen::Vector3f(0,0,0);
}

Ray::Ray(Eigen::Vector3f d)
{
	direction = d;
	eye = Eigen::Vector3f(0,0,0);
}

Ray::Ray(Eigen::Vector3f e, Eigen::Vector3f d)
{
	direction = d;
	eye = e;
}

Ray::~Ray()
{
	
}
