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

void Sphere::SetMaterialToMat(Material newMat) {
	mat = newMat;
}

void Sphere::SetMaterialByNum(int colorNum) {
	Eigen::Vector3f a, s, d;
	float shine;

	switch (colorNum) {
		case 0:
			a = Eigen::Vector3f(.2, .2, .2);
			d = Eigen::Vector3f(.8, .2, .2);
			s = Eigen::Vector3f(.4, .4, .4);
			shine = 200;
			break;
		
		case 1:
			a = Eigen::Vector3f(.2, .2, .2);
			d = Eigen::Vector3f(.2, .2, .8);
			s = Eigen::Vector3f(.4, .4, .4);
			shine = 200;
			break;

		case 2:
			a = Eigen::Vector3f(.2, .2, .2);
			d = Eigen::Vector3f(.2, .8, .2);
			s = Eigen::Vector3f(.4, .4, .4);
			shine = 200;
			break;

		case 3:
			a = Eigen::Vector3f(.2, .2, .2);
			d = Eigen::Vector3f(.8, .2, .8);
			s = Eigen::Vector3f(.4, .4, .4);
			shine = 200;
			break;

		case 4:
			a = Eigen::Vector3f(.2, .2, .2);
			d = Eigen::Vector3f(.2, .8, .8);
			s = Eigen::Vector3f(.4, .4, .4);
			shine = 200;
			break;

		case 5:
			a = Eigen::Vector3f(.2, .2, .2);
			d = Eigen::Vector3f(.8, .8, .2);
			s = Eigen::Vector3f(.4, .4, .4);
			shine = 200;
			break;

		default:
			a = Eigen::Vector3f(.2, .2, .2);
			d = Eigen::Vector3f(.6, .6, .6);
			s = Eigen::Vector3f(.6, .6, .6);
			shine = 260;
			break;
	}

	mat.ambient = a;
	mat.diffuse = d;
	mat.specular = s;
	mat.shine = shine;
}

void Sphere::SetMaterial(string colorName) {
	if(colorName == "red") {
		SetMaterialByNum(0);
	}
	
	else if(colorName == "blue") {
		SetMaterialByNum(1);
	}

	else if(colorName == "green"){
		SetMaterialByNum(2);
	}

	else if(colorName == "purple"){
		SetMaterialByNum(3);
	}

	else if(colorName == "teal"){
		SetMaterialByNum(4);
	}

	else if(colorName == "orange"){
		SetMaterialByNum(5);
	}

	else {
		cout << "ERROR! " << colorName << " is not a valid color! Here is teal, the color you should have picked" << endl;
		SetMaterialByNum(4);
	}
}


float Sphere::checkHit(Ray ray) {
	return checkHit(ray.eye, ray.direction);
}

Eigen::Vector3f CalcT(Eigen::Vector3f eye, Eigen::Vector3f dir, Eigen::Vector3f center, float radius) {
	double discriminate;
	Eigen::Vector3f dist = eye-center;

	discriminate = (dir.dot(dist)*dir.dot(dist)) 
					- (dir.dot(dir)) * (dist.dot(dist)) 
					- radius*radius;

	if (discriminate < 0) {
		return Eigen::Vector3f(0,0,0);
	}

	double sqrtDisc = sqrt(discriminate);

	double plusOp = -dir.dot(dist) + sqrtDisc;
	plusOp = plusOp / (dir.dot(dir));

	if (discriminate == 0) {
		return Eigen::Vector3f(1, plusOp, 0);
	}

	double minOp = -dir.dot(dist) - sqrtDisc;
	plusOp = plusOp / (dir.dot(dir));

	return Eigen::Vector3f(2, plusOp, minOp);
}

float Sphere::checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir) {
	Eigen::Vector3f dist = eye-center;

	double A = dir.dot(dir);
	double B = (2*dir).dot(dist);
	double C = (dist).dot(dist) - radius*radius;

	Eigen::Vector3f quad = /*CalcT(eye, dir, center, radius);*/ QuadraticFormula(A, B, C);

	if (quad(0) == 0) {
		//SHOULD BE AN ERROR
		return 0;
	}

	if (quad(0) == 1) {
		return quad(1);
	}

	if (abs(quad(1)) <= abs(quad(2))) {
		return quad(1);
	} else {
		return quad(2);
	}
}

Eigen::Vector3f QuadraticFormula(double A, double B, double C) {
	double discriminate = B*B - 4*A*C;

	if (discriminate < 0) {
		return Eigen::Vector3f(0,0,0);
	}

	double sqrtDisc = sqrt(discriminate);

	float plusOp = (-B + sqrtDisc)/(2*A);

	if (discriminate == 0) {
		return Eigen::Vector3f(1, plusOp, 0);
	}

	float minOp = (-B - sqrtDisc)/(2*A);

	return Eigen::Vector3f(2, plusOp, minOp);
}