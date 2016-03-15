#pragma  once
#ifndef __Scene__
#define __Scene__

#include <vector>
#include "Shape.hpp"
#include "Triangle.hpp"
#include "Sphere.hpp"
#include "Plane.hpp"
#include "types.h"
#include "Pixel.hpp"
#include "Vector3f.h"
#include "VectorMath.h"

typedef struct hit_struct {
   Shape *hitShape;
   double t;
   bool isHit;
} hit_t;

class Scene
{
	public:
		Scene();
		~Scene();

		Camera camera;
		std::vector<Light> lights;
		hit_t checkHit(Ray testRay);
		hit_t checkHit(Ray testRay, Shape *exclude);

		std::vector<Triangle> triangles;
		std::vector<Sphere> spheres;
		std::vector<Plane> planes;
		std::vector<Triangle> triangles_d;
		std::vector<Sphere> spheres_d;
		std::vector<Plane> planes_d;

		Pixel ComputeLighting(Ray laser, hit_t hitResult, bool print);
	private:
};

#endif
