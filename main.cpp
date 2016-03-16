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
   Ray laser;

   #ifdef DEBUG
   int noHitCount = 0;
   cout << "Triangles: " << scene.triangles.size() << endl;
   cout << "Spheres: " << scene.spheres.size() << endl;
   cout << "Planes: " << scene.planes.size() << endl;
   cout << "Lights: " << scene.lights.size() << endl;
   #endif

   scene.setupCudaMem(pic.pixels.size()*sizeof(Pixel));

   renderStart(width, height, backgroundCol, CameraRight, CameraUp,
               CameraPos, CameraDirection, scene.pixels_d,
               scene.lights_d, scene.lights.size(),
               scene.planes_d, scene.planes.size(),
               scene.triangles_d, scene.triangles.size(),
               scene.spheres_d, scene.spheres.size());

   scene.getCudaMem(&(pic.pixels[0]), pic.pixels.size());
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
/*
   Vector3f v = Vector3f(3.5, 4, 8);
   cout << "V: " << v[0] << ", " << v[1] << ", " << v[2] << endl;
   Vector3f w = v * 2.0f;
   cout << "W: " << w[0] << ", " << w[1] << ", " << w[2] << endl;
   cout << "V: " << v[0] << ", " << v[1] << ", " << v[2] << endl;
   Vector3f nv = normalize(v);
   Vector3f nw = normalize(w);
   cout << "NW: " << nw[0] << ", " << nw[1] << ", " << nw[2] << endl;
   cout << "NV: " << nv[0] << ", " << nv[1] << ", " << nv[2] << endl;
   cout << "W: " << w[0] << ", " << w[1] << ", " << w[2] << endl;
   cout << "V: " << v[0] << ", " << v[1] << ", " << v[2] << endl;
   cout << "Mag(v): " << magnitude(v) << endl;
   cout << "Mag(w): " << magnitude(w) << endl;
   Vector3f wminv = w - v;
   Vector3f vminw = v - w;
   cout << "w-v: " << wminv[0] << ", " << wminv[1] << ", " << wminv[2] << endl;
   cout << "v-w: " << vminw[0] << ", " << vminw[1] << ", " << vminw[2] << endl;
   Vector3f add = v + w;
   cout << "v+w: " << add[0] << ", " << add[1] << ", " << add[2] << endl;
   exit(0);
*/
	//Scene starts here
	loadScene();
	SetupPicture();
	PrintPicture();

	return 0;
}
