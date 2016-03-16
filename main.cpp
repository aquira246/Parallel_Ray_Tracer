#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
//#include <memory>

#include "Picture.hpp"
#include "Scene.hpp"
#include "Sphere.hpp"
#include "Sphere.hpp"
#include "Triangle.hpp"
#include "Ray.hpp"
#include "Shape.hpp"
#include "Parse.hpp"
#include "VectorMath.h"
#include "Vector3f.h"

#include <vector>
#include <math.h>

#include <cuda_runtime.h>

using namespace std;

Vector3f backgroundCol;
Picture pic;
Scene scene;

Vector3f Up = Vector3f(0,1,0);
Vector3f CameraPos, CameraDirection, CameraRight, CameraUp;

bool USE_DIRECTION = false;

int width = 600;
int height = 600;

void InitCamera() {
	CameraPos = Vector3f(0,0,10);
	CameraDirection = normalize(Vector3f(0,0,-1));
	CameraRight = cross(CameraDirection, Up);
	CameraUp = cross(CameraRight, CameraDirection);
}

void InitCamera(Camera &camera) {
	CameraPos = camera.position;
	CameraDirection = normalize(camera.look_at - camera.position);
	CameraRight = camera.right;
	CameraUp = cross(CameraRight, CameraDirection);
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
   backgroundCol = Vector3f(0, 0, 0);
   InitCamera(scene.camera);
}

void SetupPicture() {
   #ifdef DEBUG
   cout << "Triangles: " << scene.triangles.size() << endl;
   cout << "Spheres: " << scene.spheres.size() << endl;
   cout << "Planes: " << scene.planes.size() << endl;
   cout << "Lights: " << scene.lights.size() << endl;
   #endif

   scene.setupCudaMem(pic.pixels.size()*sizeof(Pixel));

   renderStart(width, height, backgroundCol, CameraRight, CameraUp,
               CameraPos, CameraDirection, scene.pixels_d,
               scene.lights_d, (uint32_t) scene.lights.size(),
               scene.planes_d, (uint32_t) scene.planes.size(),
               scene.triangles_d, (uint32_t) scene.triangles.size(),
               scene.spheres_d, (uint32_t) scene.spheres.size());

   scene.getCudaMem(&(pic.pixels[0]), pic.pixels.size()*sizeof(Pixel));
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
