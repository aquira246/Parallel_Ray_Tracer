#include "Box.hpp"
#include "Sphere.hpp"

using namespace std;

Box::Box() {
	//center = Eigen::Vector3f(0,0,0);
	//width = 1;
	//height = 1;
	//depth = 1;
	//basePt = center - Eigen::Vector3f(width/2.f, height/2.f, depth/2.f);
	shapes.clear();
}
/*
Box::Box(Eigen::Vector3f c) {
	center = c;
	width = 1;
	height = 1;
	depth = 1;
	basePt = center - Eigen::Vector3f(width/2.f, height/2.f, depth/2.f);
	shapes.clear();
}
Box::Box(float w, float h, float d) {
	center = Eigen::Vector3f(0,0,0);
	width = w;
	height = h;
	depth = d;
	basePt = center - Eigen::Vector3f(width/2.f, height/2.f, depth/2.f);
	shapes.clear();
}
Box::Box(Eigen::Vector3f c, float w, float h, float d) {
	center = c;
	width = w;
	height = h;
	depth = d;
	basePt = center - Eigen::Vector3f(width/2.f, height/2.f, depth/2.f);
	shapes.clear();
}*/
Box::~Box() {
}

void Box::addShape(Shape *s) {
	//possibly check if the shape is in the box first?
	shapes.push_back(s);
}

hit_t Box::checkHit(Ray testRay) {
	Shape* hitShape = NULL;
	bool hit = false;
	float bestT = 100;

	hit_t ret;

	for (unsigned int i = 0; i < shapes.size(); ++i)
	{
		float t = shapes[i]->checkHit(testRay.eye, testRay.direction);
		if (t > 0 && t < bestT) { //MERP idk if t > 0 is right
			hitShape = shapes[i];
			bestT = t;
			hit = true;
		}
	}

	if (!hit) {
		hitShape = NULL;
	}

	ret.hitShape = hitShape;
	ret.isHit = hit;
	ret.t = bestT;

	return ret;
}
