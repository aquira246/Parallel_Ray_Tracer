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

#include <Picture.hpp>
#include <Box.hpp>
#include <Ray.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <math.h>

using namespace std;

bool keyToggles[256] = {false};

Eigen::Vector3f LightPos;
Eigen::Vector3f LightDir;
Eigen::Vector3f backgroundCol;
Picture pic;
Box boxOfSpheres;

Eigen::Vector3f Up = Eigen::Vector3f(0,1,0);
Eigen::Vector3f CameraPos, CameraDirection, CameraRight, CameraUp;

bool USE_DIRECTION = false;

int width = 400;
int height = 400;

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

	boxOfSpheres = Box(Eigen::Vector3f(0, 0, 5), 10, 10, 10);
	boxOfSpheres.addSphere(Sphere(Eigen::Vector3f(0, 0, 3), .2));
	boxOfSpheres.addSphere(Sphere(Eigen::Vector3f(.2, -1, 2), .4));
	boxOfSpheres.addSphere(Sphere(Eigen::Vector3f(.2, 1, 2), .4));
	

	boxOfSpheres.addSphere(Sphere(Eigen::Vector3f(0, 0, 1.8), .2));
	boxOfSpheres.addSphere(Sphere(Eigen::Vector3f(.5, -.5, 2), .4));

	//boxOfSpheres.addSphere(Sphere(LightPos, .5));
}

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

	if (!USE_DIRECTION) {
		isShadow = true;
		Ray shadowRay = Ray(LightPos, hitPt - LightPos);
		hit_t hitSphere = boxOfSpheres.checkHit(shadowRay);
		if (hitSphere.isHit) {
			Eigen::Vector3f shadowHit = shadowRay.eye + shadowRay.direction * hitSphere.t;
			//cout << "Hit ray: (" << hitPt (0) << ", " << hitPt (1) << ", " << hitPt (0) << ")" << endl;
			//cout << "shadow ray: (" << shadowHit (0) << ", " << shadowHit (1) << ", " << shadowHit (0) << ")" << endl;

			if (abs(shadowHit(0) - hitPt(0)) < .1 && abs(shadowHit(1) - hitPt(1)) < .1 && abs(shadowHit(2) - hitPt(2)) < .1) {
				isShadow = false;
			}
		}
	}

	if (isShadow) {
		return Pixel(hitResult.hitSphere->mat.ambient(0), hitResult.hitSphere->mat.ambient(1), hitResult.hitSphere->mat.ambient(2));
	}

	Eigen::Vector3f n = (hitPt - hitResult.hitSphere->center).normalized();

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

	Eigen::Vector3f colorD = hold * hitResult.hitSphere->mat.diffuse;

	hold = h.dot(n);
	
	if (hold < 0)
		hold = 0;

	Eigen::Vector3f colorS = pow(hold, hitResult.hitSphere->mat.shine) * hitResult.hitSphere->mat.specular;
	Eigen::Vector3f color = colorD + colorS + hitResult.hitSphere->mat.ambient;

	if (print) {
		cout << "color: " << color << endl << endl;
		cout << "ambient: " << hitResult.hitSphere->mat.ambient << endl << endl;
		cout << "diffuse: " << colorD << endl << endl;
		cout << "specular: " << colorS << endl << endl;
		cout << "normals: " << n << endl << endl;
		cout << "Hit Point " << hitPt << endl << endl << "Center " << hitResult.hitSphere->center<<endl << endl;
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
			hit_t hitSphere = boxOfSpheres.checkHit(laser);
			if (hitSphere.isHit) {
				/*if (x > width/2 - 5 && x < width/2 + 5 && y > height/2 - 5 && y < height/2 + 5)
					pic.setPixel(x, y, ComputeLighting(laser, hitSphere, true));
				else*/
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
	glutInit(&argc, argv);
	glutInitWindowSize(400, 400);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutMouseFunc(mouseGL);			// may use later for "real time" Ray tracing?
	glutMotionFunc(mouseMotionGL);	// may use later for "real time" Ray tracing?
	glutKeyboardFunc(keyboardGL);	// may use later for "real time" Ray tracing?
	loadScene();

	//Scene starts here

	SetupPicture();
	PrintPicture();

	return 0;
}
