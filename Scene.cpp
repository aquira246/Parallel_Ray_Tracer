#include "Scene.hpp"
#include "Sphere.hpp"
#include "Triangle.hpp"

using namespace std;

Scene::Scene() {
	lights.clear();
	triangles.clear();
	spheres.clear();
}

Scene::~Scene() {
	lights.clear();
	triangles.clear();
	spheres.clear();
}

hit_t Scene::checkHit(Ray testRay) {
	Shape* hitShape = NULL;
	bool hit = false;
	float bestT = 100;

	for (unsigned int i = 0; i < planes.size(); ++i)
	{
		float t = planes[i].checkHit(testRay.eye, testRay.direction);
		if (t > 0 && t < bestT) { //MERP idk if t > 0 is right
			hitShape = &(planes[i]);
			bestT = t;
			hit = true;
         #ifdef DEBUG2
         cout << "New best hit, position of shape: "
              << (*hitShape).center[0] << ", "
              << (*hitShape).center[1] << ", "
              << (*hitShape).center[2] << endl;
         #endif
		}
	}

	for (unsigned int i = 0; i < triangles.size(); ++i)
	{
		float t = triangles[i].checkHit(testRay.eye, testRay.direction);
		if (t > 0 && t < bestT) { //MERP idk if t > 0 is right
			hitShape = &(triangles[i]);
			bestT = t;
			hit = true;
         #ifdef DEBUG2
         cout << "New best hit, position of shape: "
              << (*hitShape).center[0] << ", "
              << (*hitShape).center[1] << ", "
              << (*hitShape).center[2] << endl;
         #endif
		}
	}

	for (unsigned int i = 0; i < spheres.size(); ++i)
	{
		float t = spheres[i].checkHit(testRay.eye, testRay.direction);
		if (t > 0 && t < bestT) { //MERP idk if t > 0 is right
			hitShape = &(spheres[i]);
			bestT = t;
			hit = true;
         #ifdef DEBUG2
         cout << "New best hit, position of shape: "
              << (*hitShape).center[0] << ", "
              << (*hitShape).center[1] << ", "
              << (*hitShape).center[2] << endl;
         #endif
		}
	}

	if (!hit) {
		hitShape = NULL;
	}

	hit_t ret;
	ret.hitShape = hitShape;
	ret.isHit = hit;
	ret.t = bestT;

	return ret;
}

Pixel Scene::ComputeLighting(Ray laser, hit_t hitResult, bool print) {
	Eigen::Vector3f hitPt = laser.eye + laser.direction*hitResult.t;
	bool inShadow = true;
	Eigen::Vector3f rgb = hitResult.hitShape->mat.rgb;
	Eigen::Vector3f ambient = rgb*hitResult.hitShape->mat.ambient;
	Eigen::Vector3f color= ambient;

	// calculate if the point is in a shadow. If so, we later return the pixel as all black
	for (int i = 0; i < lights.size(); ++i)
	{
		inShadow = true;
		Ray shadowRay = Ray(lights[i].location, hitPt - lights[i].location);
		hit_t hitSphere = checkHit(shadowRay);
		if (hitSphere.isHit) {
			Eigen::Vector3f shadowHit = shadowRay.eye + shadowRay.direction * hitSphere.t;

			// makes sure we are not shadowing ourselves
			if (abs(shadowHit(0) - hitPt(0)) < 0.1 && abs(shadowHit(1) - hitPt(1)) < 0.1 && abs(shadowHit(2) - hitPt(2)) < 0.1) {
				inShadow = false;
			}
		}

		if (!inShadow) {
			Eigen::Vector3f n = (hitPt - hitResult.hitShape->center).normalized();

			Eigen::Vector3f l;
			l = (lights[i].location - hitPt);
			l.normalize();
			//l = normalize(l);

			Eigen::Vector3f v = -hitPt;
			v.normalize();
			//v = normalize(v);

			Eigen::Vector3f h = l + v;
			h.normalize();
			//h = normalize(h);

			double hold = max(dot(l, n), 0.0f);

			Eigen::Vector3f colorD = hold * rgb;

			hold = max(dot(h, n), 0.0f);

			Eigen::Vector3f colorS = pow(hold, hitResult.hitShape->mat.shine) * rgb;
			Eigen::Vector3f toAdd = colorD*hitResult.hitShape->mat.diffuse + colorS*hitResult.hitShape->mat.specular;
			toAdd[0] *= lights[i].color.r;
			toAdd[1] *= lights[i].color.g;
			toAdd[2] *= lights[i].color.b;
			color = color + toAdd;
		}
	}

	return Pixel(color(0), color(1), color(2));
}
