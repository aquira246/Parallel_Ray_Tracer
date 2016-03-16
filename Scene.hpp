#pragma  once
#ifndef __Scene__
#define __Scene__

#include <vector>
#include <cuda_runtime.h>
#include <stdint.h>
#include "Shape.hpp"
#include "Triangle.hpp"
#include "Sphere.hpp"
#include "Plane.hpp"
#include "types.h"
#include "Pixel.hpp"
#include "Vector3f.h"
#include "VectorMath.h"

#define TILE_WIDTH 32

typedef struct hit_t {
   Shape *hitShape;
   double t;
   bool isHit;
   float nx; 
   float ny;
   float nz;
} hit_t;

class Scene
{
	public:
		Scene();
		~Scene();

      void setupCudaMem(int bufferSize);
      void getCudaMem(Pixel *pixels_h, int bufferSize);

		Camera camera;
		std::vector<Light> lights;
		std::vector<Triangle> triangles;
		std::vector<Sphere> spheres;
		std::vector<Plane> planes;
		Light *lights_d;
		Triangle *triangles_d;
		Sphere *spheres_d;
		Plane *planes_d;
      Pixel *pixels_d;

	private:
};

void renderStart(int width, int height,
                 Vector3f backgroundCol, Vector3f CameraRight,
                 Vector3f CameraUp, Vector3f CameraPos,
                 Vector3f CameraDirection, Pixel *pixels,
                 Light *lights, int numLights,
                 Plane *planes, int numPlanes,
                 Triangle *triangles, int numTriangles,
                 Sphere *spheres, int numSpheres);

void checkCudaErrors(int errorCode, char const *callName);

__device__ hit_t checkHit(Ray testRay, Shape *exclude,
                          Plane *planes, int numPlanes,
                          Triangle *triangles, int numTriangles,
                          Sphere *spheres, int numSpheres);

__device__ Pixel ComputeLighting(Ray laser, hit_t hitResult,
                                 Light *lights, int numLights,
                                 Plane *planes, int numPlanes,
                                 Triangle *triangles, int numTriangles,
                                 Sphere *spheres, int numSpheres);

__global__ void renderScene(float aspectRatio, int width, int height,
                            Vector3f backgroundCol, Vector3f CameraRight,
                            Vector3f CameraUp, Vector3f CameraPos,
                            Vector3f CameraDirection, Pixel *pixels,
                            Light *lights, int numLights,
                            Plane *planes, int numPlanes,
                            Triangle *triangles, int numTriangles,
                            Sphere *spheres, int numSpheres);

__device__
float plane_CheckHit(Plane *plane, Vector3f eye, Vector3f dir);
__device__
float sphere_CheckHit(Sphere *sphere, Vector3f eye, Vector3f dir);
__device__ 
float triangle_CheckHit(Triangle *tri, Vector3f eye, Vector3f dir);


#endif
