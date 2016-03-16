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
		Light *lights_d;
		hit_t checkHit(Ray testRay);
		hit_t checkHit(Ray testRay, Shape *exclude);

		std::vector<Triangle> triangles;
		std::vector<Sphere> spheres;
		std::vector<Plane> planes;
		Triangle *triangles_d;
		Sphere *spheres_d;
		Plane *planes_d;

		Pixel ComputeLighting(Ray laser, hit_t hitResult, bool print);
	private:
};

#endif

/*put in scene.hpp*/
void setupCudaData(Pixel* pixels_h, Pixel** pixels_d, int bufferSize);
__global__ void renderScene(float aspectRatio, Vector3f CameraRight, Vector3f CameraUp, Vector3f CameraPos, Vector3f CameraDirection, Pixel *pixels);

/*put in scene.cu*/
/*
 * Allocate and copy data into device memory.
 */

__device__ hit_t Scene::checkHit(Ray testRay) {
   Shape* hitShape = NULL;
   bool hit = false;
   float bestT = 10000;

   for (unsigned int i = 0; i < planes.size(); ++i)
   {
      float t = planes_d[i].checkHit(testRay.eye, testRay.direction);
      if (t > 0 && t < bestT) { //MERP idk if t > 0 is right
         hitShape = &(planes_d[i]);
         bestT = t;
         hit = true;
      }
   }

   for (unsigned int i = 0; i < triangles.size(); ++i)
   {
      float t = triangles_d[i].checkHit(testRay.eye, testRay.direction);
      if (t > 0 && t < bestT) { //MERP idk if t > 0 is right
         hitShape = &(triangles_d[i]);
         bestT = t;
         hit = true;
      }
   }

   for (unsigned int i = 0; i < spheres.size(); ++i)
   {
      float t = spheres_d[i].checkHit(testRay.eye, testRay.direction);
      if (t > 0 && t < bestT) { //MERP idk if t > 0 is right
         hitShape = &(spheres_d[i]);
         bestT = t;
         hit = true;
      }
   }

   if (!hit) {
      hitShape = NULL;
   }

   hit_t ret;
   ret.hitShape = hitShape;
   ret.isHit = hit;
   ret.t = bestT;

   return ret;
}

void setupCudaData(Pixel* pixels_h, Pixel** pixels_d, int bufferSize)
{
   /* allocate device memory */
   checkCudaErrors(cudaMalloc((void**) pixels_d, bufferSize), "cudaMalloc");
   checkCudaErrors(cudaMalloc((void**) lights_d, lights.size()), "cudaMalloc");
   checkCudaErrors(cudaMalloc((void**) triangles_d, triangles.size()), "cudaMalloc");
   checkCudaErrors(cudaMalloc((void**) spheres_d, spheres.size()), "cudaMalloc");
   checkCudaErrors(cudaMalloc((void**) planes_d, planes.size()), "cudaMalloc");

   /* copy into device memory */
   checkCudaErrors(cudaMemcpy(*pixels_d, pixels_h, bufferSize, cudaMemcpyHostToDevice), "cudaMemcpy");
   checkCudaErrors(cudaMemcpy(*lights_d, &lights[0], lights.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
   checkCudaErrors(cudaMemcpy(*triangles_d, &triangles[0], triangles.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
   checkCudaErrors(cudaMemcpy(*spheres_d, &spheres[0], spheres.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
   checkCudaErrors(cudaMemcpy(*planes_d, &planes[0], planes.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
}

__global__ void renderScene(float aspectRatio, Vector3f CameraRight, Vector3f CameraUp, Vector3f CameraPos, Vector3f CameraDirection, Pixel *pixels)
   int Row = blockIdx.y*blockDim.y + threadIdx.y;
   int Col = blockIdx.x*blockDim.x + threadIdx.x;

   float normalized_i, normalized_j;
   if(aspectRatio > 1) {
      normalized_i = ((col/(float)gridDim.x) - 0.5) * aspectRatio;
      normalized_j = (row/(float)gridDim.y) - 0.5;
   }
   else {
      normalized_i = (col/(float)gridDim.x) - 0.5;
      normalized_j = ((row/(float)gridDim.y) - 0.5) / aspectRatio;
   }

   Vector3f imagePoint = CameraRight * normalized_i + 
                         CameraUp * normalized_j +
                         CameraPos + CameraDirection;

   Vector3f ray_direction = imagePoint - CameraPos;

   laser = Ray(CameraPos, ray_direction);

   hit_t hitShape = scene.checkHit(laser);
   
   if (hitShape.isHit) {
      pixels_d[x + y*gridDim.x)] =  scene.ComputeLighting(laser, hitShape, false));
   } else {
      // not hit means we color it with a background color
      pixels_d[x + y*gridDim.x)] = Pixel(0.5, 0.5, 0.5));
   }
}

// move the pixels from the device to a Pic class when this is done