#include "Scene.hpp"
#include "Sphere.hpp"
#include "Triangle.hpp"

using namespace std;

Scene::Scene() {
	lights.clear();
	triangles.clear();
	spheres.clear();
}

Scene::~Scene() {
	lights.clear();
	triangles.clear();
	spheres.clear();
}

void Scene::setupCudaMem(int bufferSize) {
  cout << "Starting to setup cuda" << endl;

   /* allocate device memory */
   checkCudaErrors(cudaMalloc((void**) &pixels_d, bufferSize), "cudaMalloc");
   checkCudaErrors(cudaMalloc((void**) &lights_d, sizeof(Light)*lights.size()), "cudaMalloc");
   checkCudaErrors(cudaMalloc((void**) &triangles_d, sizeof(Triangle)*triangles.size()), "cudaMalloc");
   checkCudaErrors(cudaMalloc((void**) &spheres_d, sizeof(Sphere)*spheres.size()), "cudaMalloc");
   checkCudaErrors(cudaMalloc((void**) &planes_d, sizeof(Plane)*planes.size()), "cudaMalloc");

   cout << "CudaMalloc Success" << endl;

   /* copy into device memory */
   checkCudaErrors(cudaMemcpy(lights_d, &(lights[0]), sizeof(Light)*lights.size(),
                   cudaMemcpyHostToDevice), "cudaMemcpy");
   checkCudaErrors(cudaMemcpy(triangles_d, &(triangles[0]), sizeof(Triangle)*triangles.size(),
                   cudaMemcpyHostToDevice), "cudaMemcpy");
   checkCudaErrors(cudaMemcpy(spheres_d, &(spheres[0]), sizeof(Sphere)*spheres.size(),
                   cudaMemcpyHostToDevice), "cudaMemcpy");
   checkCudaErrors(cudaMemcpy(planes_d, &(planes[0]), sizeof(Plane)*planes.size(),
                   cudaMemcpyHostToDevice), "cudaMemcpy");

   cout << "Finished setting up cuda" << endl;
}

void Scene::getCudaMem(Pixel *pixels_h, int bufferSize) {
   checkCudaErrors(cudaMemcpy(pixels_h, pixels_d, bufferSize,
                   cudaMemcpyDeviceToHost), "cudaMemcpy");
   //TODO memcopy back for pixels is correct???
   //TODO free gpu mem
   cout << "CudaMem Complete" << endl;
}

void renderStart(int width, int height,
                 Vector3f backgroundCol, Vector3f CameraRight,
                 Vector3f CameraUp, Vector3f CameraPos,
                 Vector3f CameraDirection, Pixel *pixels,
                 Light *lights, int numLights,
                 Plane *planes, int numPlanes,
                 Triangle *triangles, int numTriangles,
                 Sphere *spheres, int numSpheres)
{
  cout << "Starting render" << endl;
   float aspectRatio = (float) width / height;
   //int dim = ceil((double) width * height / TILE_WIDTH);
   dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
   dim3 dimGrid(ceil((double) width / TILE_WIDTH),
                ceil((double) height / TILE_WIDTH));

   renderScene<<<dimGrid, dimBlock>>>(aspectRatio, width, height,
                                      backgroundCol, CameraRight,
                                      CameraUp, CameraPos,
                                      CameraDirection, pixels,
                                      lights, numLights,
                                      planes, numPlanes,
                                      triangles, numTriangles,
                                      spheres, numSpheres);

   cout << "Finished render start" << endl;
}

/*
 * Check for errors on cuda calls and exit if one is thrown.
 */
void checkCudaErrors(int errorCode, char const *callName) {
   if(errorCode > 0) {
      printf("Cuda Error: %s\n", callName);
      exit(EXIT_FAILURE);
   }
}

__device__ hit_t checkHit(Ray testRay, Shape *exclude,
                          Plane *planes, int numPlanes,
                          Triangle *triangles, int numTriangles,
                          Sphere *spheres, int numSpheres)
{
	Shape* hitShape = NULL;
	bool hit = false;
	float bestT = 10000;

	for (unsigned int i = 0; i < numPlanes; ++i)
	{
      if(&(planes[i]) != exclude) {
		   float t = planes[i].checkHit(testRay.eye, testRay.direction);
		   if (t > 0 && t < bestT) {
			   hitShape = &(planes[i]);
			   bestT = t;
   			hit = true;
         }
		}
	}

	for (unsigned int i = 0; i < numTriangles; ++i)
	{
      if(&(triangles[i]) != exclude) {
   		float t = triangles[i].checkHit(testRay.eye, testRay.direction);
	   	if (t > 0 && t < bestT) {
		   	hitShape = &(triangles[i]);
			   bestT = t;
   			hit = true;
         }
		}
	}

	for (unsigned int i = 0; i < numSpheres; ++i)
	{
      if(&(spheres[i]) != exclude) {
   		float t = spheres[i].checkHit(testRay.eye, testRay.direction);
	   	if (t > 0 && t < bestT) {
		   	hitShape = &(spheres[i]);
			   bestT = t;
   			hit = true;
         }
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

__device__ Pixel ComputeLighting(Ray laser, hit_t hitResult,
                                 Light *lights, int numLights,
                                 Plane *planes, int numPlanes,
                                 Triangle *triangles, int numTriangles,
                                 Sphere *spheres, int numSpheres)
{
	Vector3f hitPt = laser.eye + laser.direction*hitResult.t;
	Vector3f viewVec = -laser.direction;
	Vector3f rgb = hitResult.hitShape->mat.rgb;
	Vector3f ambient = rgb*hitResult.hitShape->mat.ambient;
   Vector3f n = hitResult.hitShape->GetNormal(hitPt);
	Vector3f color;
	bool inShadow;

	// calculate if the point is in a shadow. If so, we later return the pixel as all black
	for (int i = 0; i < numLights; ++i)
	{
		inShadow = false;
		Vector3f shadowDir = normalize(lights[i].location - hitPt);
	   Vector3f l = shadowDir;//normalize(lights[i].location - hitPt);
		Ray shadowRay = Ray(hitPt, shadowDir);
		hit_t shadowHit = checkHit(shadowRay, hitResult.hitShape,
                                 planes, numPlanes,
                                 triangles, numTriangles,
                                 spheres, numSpheres);

		if (shadowHit.isHit) {
			if (shadowHit.hitShape != hitResult.hitShape)
				inShadow = true;
		}

      if (!inShadow) {
         Vector3f r = -l + n * 2 * dot(n,l);
         r = normalize(r);

         float specMult = max(dot(viewVec, r), 0.0f);
         specMult = pow(specMult, hitResult.hitShape->mat.shine);
         
         Vector3f colorS = rgb * specMult;

			float hold = min(max(dot(l, n), 0.0f), 1.0f);
			Vector3f colorD = rgb * hold;

			Vector3f toAdd = colorD * hitResult.hitShape->mat.diffuse
                               + colorS * hitResult.hitShape->mat.specular;
         //spec + diffuse setup
			toAdd[0] *= lights[i].color.r;
			toAdd[1] *= lights[i].color.g;
			toAdd[2] *= lights[i].color.b;
         //actually add spec + diffuse
			color = color + toAdd;
		}
      //ambient addition
	   color[0] += ambient[0] * lights[i].color.r;
	   color[1] += ambient[1] * lights[i].color.g;
	   color[2] += ambient[2] * lights[i].color.b;
      //make sure in range still
      color[0] = min(max(color[0],0.0f),1.0f);
      color[1] = min(max(color[1],0.0f),1.0f);
      color[2] = min(max(color[2],0.0f),1.0f);
	}

	return Pixel(color[0], color[1], color[2]);
}

__global__ void renderScene(float aspectRatio, int width, int height,
                            Vector3f backgroundCol, Vector3f CameraRight,
                            Vector3f CameraUp, Vector3f CameraPos,
                            Vector3f CameraDirection, Pixel *pixels,
                            Light *lights, int numLights,
                            Plane *planes, int numPlanes,
                            Triangle *triangles, int numTriangles,
                            Sphere *spheres, int numSpheres)
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int pixelNum = row * width + col;
   //printf("starting to render scene at (%d, %d)\n", row, col);
   if(pixelNum < width * height) { 

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

   Ray laser = Ray(CameraPos, ray_direction);

   // no shape to exclude so use NULL 
   hit_t hitShape = checkHit(laser, NULL,
                             planes, numPlanes,
                             triangles, numTriangles,
                             spheres, numSpheres);
   
   //printf("actually hit shape\n");
   if (hitShape.isHit) {
      pixels[pixelNum] = ComputeLighting(laser, hitShape,
                                                  lights, numLights,
                                                  planes, numPlanes,
                                                  triangles, numTriangles,
                                                  spheres, numSpheres);
   } else {
      // not hit means we color it with a background color
      pixels[pixelNum] = Pixel(backgroundCol[0],
                                            backgroundCol[1],
                                            backgroundCol[2]);
   }
   }
}

