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

//bool keyToggles[256] = {false};

//Eigen::Vector3f LightPos;
//Eigen::Vector3f LightDir;
Eigen::Vector3f backgroundCol;
Picture pic;
Scene scene;
//std::vector<Sphere> allSpheres;
//std::vector<Triangle> allTriangles;

Eigen::Vector3f Up = Eigen::Vector3f(0,1,0);
Eigen::Vector3f CameraPos, CameraDirection, CameraRight, CameraUp;

bool USE_DIRECTION = false;

int width = 1600;
int height = 1600;

void InitCamera() {
	CameraPos = Eigen::Vector3f(0,0,0);
	CameraDirection = Eigen::Vector3f(0,0,1);
	CameraRight = cross(CameraDirection, Up);
	CameraUp = cross(CameraRight, CameraDirection);
}

void InitCamera(Camera camera) {
	CameraPos = camera.position;
	CameraDirection = camera.direction;
	CameraRight = camera.right;
	CameraUp = camera.up;
}

void loadScene()
{
	pic = Picture(width, height);
	backgroundCol = Eigen::Vector3f(0,0,0);
	InitCamera(scene.camera);   
}

Ray ComputeCameraRay(int i, int j) {
	double normalized_i = (i/(float)pic.width) - .5;
	double normalized_j = (j/(float)pic.height) - .5;

	Eigen::Vector3f imagePoint = normalized_i * CameraRight + 
								normalized_j * CameraUp +
								CameraPos + CameraDirection;
	
	Eigen::Vector3f ray_direction = imagePoint - CameraPos;

	return Ray(CameraPos, ray_direction);
}

void SetupPicture() {
	Ray laser;

	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < height; ++y)
		{
			laser = ComputeCameraRay(x, y);
			hit_t hitSphere = scene.checkHit(laser);
			if (hitSphere.isHit) {
					pic.setPixel(x, y, scene.ComputeLighting(laser, hitSphere, false));
			} else {
				pic.setPixel(x, y, Pixel(.5, .5, .5));
			}
		}
	}
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
