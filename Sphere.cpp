#include <Sphere.hpp>

using namespace std;

Sphere::Sphere() {
	SetMaterialByNum(rand() % NUM_MATS);
	center = Eigen::Vector3f(0,0,0);
	radius = 1;
}
Sphere::Sphere(Eigen::Vector3f c) {
	SetMaterialByNum(rand() % NUM_MATS);
	center = c;
	radius = 1;
}
Sphere::Sphere(float r){
	SetMaterialByNum(rand() % NUM_MATS);
	center = Eigen::Vector3f(0,0,0);
	radius = r;
}
Sphere::Sphere(Eigen::Vector3f c, float r){
	SetMaterialByNum(rand() % NUM_MATS);
	center = c;
	radius = r;
}
Sphere::~Sphere(){

}

float Sphere::checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir) {
	Eigen::Vector3f dist = eye - center;

	double A = dir.dot(dir);
	double B = (2*dir).dot(dist);
	double C = (dist).dot(dist) - radius*radius;

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
