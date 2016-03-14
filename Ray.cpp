#include <Ray.hpp>

Ray::Ray()
{
	direction = Vector3f(0,0,1);
	eye = Vector3f(0,0,0);
}

Ray::Ray(Vector3f d)
{
	direction = d;
	eye = Vector3f(0,0,0);
}

Ray::Ray(Vector3f e, Vector3f d)
{
	direction = d;
	eye = e;
}

Ray::~Ray()
{
	
}
