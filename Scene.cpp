#include "Scene.hpp"
#include "Sphere.hpp"

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

	hit_t ret;

	for (unsigned int i = 0; i < triangles.size(); ++i)
	{
		float t = triangles[i].checkHit(testRay.eye, testRay.direction);
		if (t > 0 && t < bestT) { //MERP idk if t > 0 is right
			hitShape = &(triangles[i]);
			bestT = t;
			hit = true;
		}
	}

	for (unsigned int i = 0; i < spheres.size(); ++i)
	{
		float t = spheres[i].checkHit(testRay.eye, testRay.direction);
		if (t > 0 && t < bestT) { //MERP idk if t > 0 is right
			hitShape = &(spheres[i]);
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

Pixel Scene::ComputeLighting(Ray laser, hit_t hitResult, bool print) {
	Eigen::Vector3f hitPt = laser.eye + laser.direction*hitResult.t;
	bool isShadow = false;
	Eigen::Vector3f color= Eigen::Vector3f(0,0,0);
	Eigen::Vector3f rgb = hitResult.hitShape->mat.rgb;
	Eigen::Vector3f ambient = rgb*hitResult.hitShape->mat.ambient;

	// calculate if the point is in a shadow. If so, we later return the pixel as all black
	for (int i = 0; i < lights.size(); ++i)
	{
		isShadow = true;
		Ray shadowRay = Ray(lights[i].location, hitPt - lights[i].location);
		hit_t hitSphere = checkHit(shadowRay);
		if (hitSphere.isHit) {
			Eigen::Vector3f shadowHit = shadowRay.eye + shadowRay.direction * hitSphere.t;

			// makes sure we are not shadowing ourselves
			if (abs(shadowHit(0) - hitPt(0)) < .1 && abs(shadowHit(1) - hitPt(1)) < .1 && abs(shadowHit(2) - hitPt(2)) < .1) {
				isShadow = false;
			}
		}

		if (!isShadow) {
			Eigen::Vector3f n = (hitPt - hitResult.hitShape->center).normalized();

			Eigen::Vector3f l;
			l = (lights[i].location - hitPt);
			l = normalize(l);

			Eigen::Vector3f v = -hitPt;
			v = normalize(v);

			Eigen::Vector3f h = l + v;
			h = normalize(h);

			double hold = max(dot(l, n), 0.0f);

			Eigen::Vector3f colorD = hold * rgb;

			hold = max(dot(h, n), 0.0f);

			Eigen::Vector3f colorS = pow(hold, hitResult.hitShape->mat.shine) * rgb;
			color = color + 
					(colorD*hitResult.hitShape->mat.diffuse 
					+ colorS*hitResult.hitShape->mat.specular);
		}
	}
	color += Eigen::Vector3f(ambient(0), ambient(1), ambient(2));

	// if (print) {
	// 	cout << "color: " << color << endl << endl;
	// 	cout << "ambient: " << ambient << endl << endl;
	// 	cout << "diffuse: " << colorD << endl << endl;
	// 	cout << "specular: " << colorS << endl << endl;
	// 	cout << "normals: " << n << endl << endl;
	// 	cout << "Hit Point " << hitPt << endl << endl << "Center " << hitResult.hitShape->center<<endl << endl;
	// }

	return Pixel(color(0), color(1), color(2));
}