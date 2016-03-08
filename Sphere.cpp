#include "Sphere.hpp"

using namespace std;

Sphere::Sphere() {
	//SetMaterialByNum(rand() % NUM_MATS);
	center = Eigen::Vector3f(0,0,0);
	radius = 1.0f;
}
Sphere::Sphere(Eigen::Vector3f c) {
	//SetMaterialByNum(rand() % NUM_MATS);
	center = c;
	radius = 1.0f;
}
Sphere::Sphere(float r){
	//SetMaterialByNum(rand() % NUM_MATS);
	center = Eigen::Vector3f(0,0,0);
	radius = r;
}
Sphere::Sphere(Eigen::Vector3f c, float r){
	//SetMaterialByNum(rand() % NUM_MATS);
	center = c;
	radius = r;
}
Sphere::~Sphere(){

}

float Sphere::checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir) {
	Eigen::Vector3f dist = eye - center;

	double A = dot(dir, dir);
	double B = dot((2*dir), dist);
	double C = dot(dist, dist) - radius*radius;

	Eigen::Vector3f quad = QuadraticFormula(A, B, C);

	if (quad(0) == 0) {
		//SHOULD BE AN ERROR
		return 0;
	}

	if (quad(0) == 1) {
		return quad(1);
	}

	if (fabs(quad(1)) <= fabs(quad(2))) {
		return quad(1);
	} else {
		return quad(2);
	}
}
