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
#include "Box.hpp"
#include "Sphere.hpp"
#include "Sphere.hpp"
#include "Triangle.hpp"
#include "Ray.hpp"
#include "Shape.hpp"
#include "Parse.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <math.h>

#include <cuda_runtime.h>

using namespace std;

bool keyToggles[256] = {false};

Eigen::Vector3f LightPos;
Eigen::Vector3f LightDir;
Eigen::Vector3f backgroundCol;
Picture pic;
Box boxOfShapes;
std::vector<Sphere> allSpheres;
std::vector<Triangle> allTriangles;

Eigen::Vector3f Up = Eigen::Vector3f(0,1,0);
Eigen::Vector3f CameraPos, CameraDirection, CameraRight, CameraUp;

bool USE_DIRECTION = false;

int width = 1600;
int height = 1600;

void InitCamera() {
	LightPos = Eigen::Vector3f(-1, 1, 1.5);
	LightDir = Eigen::Vector3f(0, 0, -1);

	CameraPos = Eigen::Vector3f(0,0,0);
	CameraDirection = Eigen::Vector3f(0,0,1);
	CameraRight = CameraDirection.cross(Up);
	CameraUp = CameraRight.cross(CameraDirection);
}

void loadScene()
{
	pic = Picture(width, height);
	backgroundCol = Eigen::Vector3f(0,0,0);
	InitCamera();

	boxOfShapes = Box(Eigen::Vector3f(0, 0, 5), 10, 10, 10);

	allSpheres.push_back(Sphere(Eigen::Vector3f(0, 0, 3), .2));
	allSpheres.push_back(Sphere(Eigen::Vector3f(.2, -1, 2), .4));
	allSpheres.push_back(Sphere(Eigen::Vector3f(.2, 1, 2), .4));
	allSpheres.push_back(Sphere(Eigen::Vector3f(0, 0, 1.8), .2));
	allSpheres.push_back(Sphere(Eigen::Vector3f(.5, -.5, 2), .4));

	// allTriangles.push_back(Triangle(Eigen::Vector3f(-5, -3, 2), Eigen::Vector3f(.5, -3, 2), Eigen::Vector3f(2.5, 2, 2)));

	for (unsigned int i = 0; i < allSpheres.size(); ++i)
	{
		boxOfShapes.addShape(&allSpheres[i]);
	}

	for (unsigned int i = 0; i < allTriangles.size(); ++i)
	{
		boxOfShapes.addShape(&allTriangles[i]);
	}
}
/*
void mouseGL(int button, int state, int x, int y)
{
	int modifier = glutGetModifiers();
	bool shift = modifier & GLUT_ACTIVE_SHIFT;
	bool ctrl  = modifier & GLUT_ACTIVE_CTRL;
	bool alt   = modifier & GLUT_ACTIVE_ALT;
}

void mouseMotionGL(int x, int y)
{

}

void keyboardGL(unsigned char key, int x, int y)
{
	if (key != 'w' && key != 'a' && key != 's' && key != 'd' && key != 'q' && key != 'e') {
		keyToggles[key] = !keyToggles[key];
	}

	switch(key) {
		case 27:
			// ESCAPE
			exit(0);
			break;
		case 'w':
			LightPos += Eigen::Vector3f(0, 1, 0);
			break;
		case 's':
			LightPos += Eigen::Vector3f(0, -1, 0);
			break;
		case 'd':
			LightPos += Eigen::Vector3f(1, 0, 0);
			break;
		case 'a':
			LightPos += Eigen::Vector3f(-1, 0, 0);
			break;
		case 'q':
			LightPos += Eigen::Vector3f(0, 0, 1);
			break;
		case 'e':
			LightPos += Eigen::Vector3f(0, 0, -1);
			break;

	}
}
*/
Ray ComputeCameraRay(int i, int j) {
	double normalized_i = (i/(float)pic.width) - .5;
	double normalized_j = (j/(float)pic.height) - .5;

	Eigen::Vector3f imagePoint = normalized_i * CameraRight + 
								normalized_j *CameraUp +
								CameraPos + CameraDirection;
	
	Eigen::Vector3f ray_direction = imagePoint - CameraPos;

	return Ray(CameraPos, ray_direction);
}

Pixel ComputeLighting(Ray laser, hit_t hitResult, bool print) {
	Eigen::Vector3f hitPt = laser.eye + laser.direction*hitResult.t;
	bool isShadow = false;

	// calculate if the point is in a shadow. If so, we later return the pixel as all black
	if (!USE_DIRECTION) {
		isShadow = true;
		Ray shadowRay = Ray(LightPos, hitPt - LightPos);
		hit_t hitSphere = boxOfShapes.checkHit(shadowRay);
		if (hitSphere.isHit) {
			Eigen::Vector3f shadowHit = shadowRay.eye + shadowRay.direction * hitSphere.t;

			// makes sure we are not shadowing ourselves
			if (abs(shadowHit(0) - hitPt(0)) < .1 && abs(shadowHit(1) - hitPt(1)) < .1 && abs(shadowHit(2) - hitPt(2)) < .1) {
				isShadow = false;
			}
		}
	}

	Eigen::Vector3f rgb = hitResult.hitShape->mat.rgb;
	Eigen::Vector3f ambient = rgb*hitResult.hitShape->mat.ambient;
	if (isShadow) {
		return Pixel(ambient(0), ambient(1), ambient(2));
	}

	Eigen::Vector3f n = (hitPt - hitResult.hitShape->center).normalized();

	Eigen::Vector3f l;
	if (!USE_DIRECTION)
		l = (LightPos - hitPt);
	else
		l = LightDir;

	l.normalize();

	Eigen::Vector3f v = -hitPt;
	v.normalize();

	Eigen::Vector3f h = l + v;
	h.normalize();

	double hold;
	hold = l.dot(n);

	if (hold < 0)
		hold = 0;

	Eigen::Vector3f colorD = hold * rgb;

	hold = h.dot(n);
	
	if (hold < 0)
		hold = 0;

	Eigen::Vector3f colorS = pow(hold, hitResult.hitShape->mat.shine) * rgb;
	Eigen::Vector3f color = colorD*hitResult.hitShape->mat.diffuse 
							+ colorS*hitResult.hitShape->mat.specular 
							+ ambient;

	if (print) {
		cout << "color: " << color << endl << endl;
		cout << "ambient: " << ambient << endl << endl;
		cout << "diffuse: " << colorD << endl << endl;
		cout << "specular: " << colorS << endl << endl;
		cout << "normals: " << n << endl << endl;
		cout << "Hit Point " << hitPt << endl << endl << "Center " << hitResult.hitShape->center<<endl << endl;
	}

	return Pixel(color(0), color(1), color(2));
}

void SetupPicture() {
	Ray laser;

	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < height; ++y)
		{
			laser = ComputeCameraRay(x, y);
			hit_t hitSphere = boxOfShapes.checkHit(laser);
			if (hitSphere.isHit) {
					pic.setPixel(x, y, ComputeLighting(laser, hitSphere, false));
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
 
   if(argc < 2) {
      cout << "Usage: rt <input_scene.pov>" << endl;
      exit(EXIT_FAILURE);
   }

   infile = fopen(argv[1], "r");
   if(infile) {
      cout << Parse(infile) << " objects parsed from scene file" << endl;
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
