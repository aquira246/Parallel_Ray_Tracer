#ifdef __APPLE__
#include <GLUT/glut.h>
#endif
#ifdef __unix__
#include <GL/glut.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <memory>

#include "Picture.hpp"
#include "Scene.hpp"
#include "Sphere.hpp"
#include "Sphere.hpp"
#include "Triangle.hpp"
#include "Ray.hpp"
#include "Shape.hpp"
#include "Parse.hpp"
#include "VectorMath.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <math.h>

#include <cuda_runtime.h>

using namespace std;

Eigen::Vector3f backgroundCol;
Picture pic;
Scene scene;

Eigen::Vector3f Up = Eigen::Vector3f(0,1,0);
Eigen::Vector3f CameraPos, CameraDirection, CameraRight, CameraUp;

bool USE_DIRECTION = false;

int width = 1200;
int height = 800;
float aspectRatio;

void InitCamera() {
	CameraPos = Eigen::Vector3f(0,0,10);
	CameraDirection = normalize(Eigen::Vector3f(0,0,-1));
	CameraRight = cross(CameraDirection, Up);
	CameraUp = cross(CameraRight, CameraDirection);
}

void InitCamera(Camera &camera) {
	CameraPos = camera.position;
	CameraDirection = normalize(camera.look_at - camera.position);
	CameraRight = camera.right;
	CameraUp = camera.up;
   #ifdef DEBUG
   cout << "Camera:\n" << "\tPOSITION: "
        << CameraPos[0] << ", " << CameraPos[1] << ", " << CameraPos[2] << endl
        << "\tDIRECTION: "
        << CameraDirection[0] << ", " << CameraDirection[1] << ", "
        << CameraDirection[2] << endl
        << "\tRIGHT: "
        << CameraRight[0] << ", " << CameraRight[1] << ", "
        << CameraRight[2] << endl
        << "\tUP: "
        << CameraUp[0] << ", " << CameraUp[1] << ", " << CameraUp[2] << endl;
   #endif
}

void loadScene()
{
   pic = Picture(width, height);
   aspectRatio = (double) width / height;
   backgroundCol = Eigen::Vector3f(0.5,0.5,0.5);
   InitCamera(scene.camera);
}

Ray ComputeCameraRay(int i, int j) {
   float normalized_i, normalized_j;
   if(aspectRatio > 1) {
      normalized_i = ((i/(float)pic.width) - 0.5) * aspectRatio;
      normalized_j = (j/(float)pic.height) - 0.5;
   }
   else {
      normalized_i = (i/(float)pic.width) - 0.5;
      normalized_j = ((j/(float)pic.height) - 0.5) / aspectRatio;
   }

   Eigen::Vector3f imagePoint = normalized_i * CameraRight + 
								normalized_j * CameraUp +
								CameraPos + CameraDirection;

   Eigen::Vector3f ray_direction = imagePoint - CameraPos;

   return Ray(CameraPos, ray_direction);
}

void SetupPicture() {
   Ray laser;

   #ifdef DEBUG
   int noHitCount = 0;
   cout << "Triangles: " << scene.triangles.size() << endl;
   cout << "Spheres: " << scene.spheres.size() << endl;
   cout << "Planes: " << scene.planes.size() << endl;
   cout << "Lights: " << scene.lights.size() << endl;
   #endif

   for (int x = 0; x < width; ++x)
   {
      for (int y = 0; y < height; ++y)
      {
         laser = ComputeCameraRay(x, y);
         hit_t hitShape = scene.checkHit(laser);
         if (hitShape.isHit) {
            pic.setPixel(x, y, scene.ComputeLighting(laser, hitShape, false));
			} else {
            pic.setPixel(x, y, Pixel(backgroundCol[0], backgroundCol[1], backgroundCol[2]));
            #ifdef DEBUG
            ++noHitCount;
            #endif
         }
         #ifdef DEBUG
         cout << "Pixel: " << x * height + y << "\r";
         #endif
      }
   }

   #ifdef DEBUG
   cout << "Rays with no hit: " << noHitCount << endl;
   #endif
}

void PrintPicture() {
	pic.Print("results.tga");
}

int main(int argc, char **argv)
{
   FILE* infile;
   scene = Scene();
 
   if(argc < 2) {
      cout << "Usage: rt <input_scene.pov>" << endl;
      exit(EXIT_FAILURE);
   }

   infile = fopen(argv[1], "r");
   if(infile) {
      cout << Parse(infile, scene) << " objects parsed from scene file" << endl;
   }
   else {
      perror("fopen");
      exit(EXIT_FAILURE);
   }

	//Scene starts here
	loadScene();
	SetupPicture();
	PrintPicture();

	return 0;
}
