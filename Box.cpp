#include <Box.hpp>

using namespace std;

Box::Box() {
	center = Eigen::Vector3f(0,0,0);
	width = 1;
	height = 1;
	depth = 1;
	basePt = center - Eigen::Vector3f(width/2.f, height/2.f, depth/2.f);
	spheres.clear();
}
Box::Box(Eigen::Vector3f c) {
	center = c;
	width = 1;
	height = 1;
	depth = 1;
	basePt = center - Eigen::Vector3f(width/2.f, height/2.f, depth/2.f);
	spheres.clear();
}
Box::Box(float w, float h, float d) {
	center = Eigen::Vector3f(0,0,0);
	width = w;
	height = h;
	depth = d;
	basePt = center - Eigen::Vector3f(width/2.f, height/2.f, depth/2.f);
	spheres.clear();
}
Box::Box(Eigen::Vector3f c, float w, float h, float d) {
	center = c;
	width = w;
	height = h;
	depth = d;
	basePt = center - Eigen::Vector3f(width/2.f, height/2.f, depth/2.f);
	spheres.clear();
}
Box::~Box() {
}

void Box::addSphere(Eigen::Vector3f center, float radius) {
	addSphere(Sphere(center, radius));
}

void Box::addSphere(Sphere s) {
	//possibly check if the sphere is in the box first?
	spheres.push_back(s);
}

hit_t Box::checkHit(Ray testRay) {
	Sphere * hitSphere;
	bool hit = false;
	float bestT = 100;

	hit_t ret;

	for (int i = 0; i < spheres.size(); ++i)
	{
		float t = spheres[i].checkHit(testRay);
		if (t > 0 && t < bestT) { //MERP idk if t > 0 is right
			hitSphere = &spheres[i];
			bestT = t;
			hit = true;
		}
	}

	if (!hit) {
		hitSphere = NULL;
	}

	ret.hitSphere = hitSphere;
	ret.isHit = hit;
	ret.t = bestT;

	return ret;
}