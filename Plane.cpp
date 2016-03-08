#include "Plane.hpp"

using namespace std;

constexpr float kEpsilon = 1e-8; 

Plane::Plane() {
    //SetMaterialByNum(rand() % NUM_MATS);
    center = Eigen::Vector3f(0,0,0);
    normal = Eigen::Vector3f(0,0,-1);
    radius = 1.0f;
}

Plane::Plane(Eigen::Vector3f c, Eigen::Vector3f n, float r) {
    //SetMaterialByNum(rand() % NUM_MATS);
    center = c;
    normal = n;
    radius = r;
}

Plane::~Plane(){

}

float Plane::checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir) {
    
    float t = -1;

    // assuming vectors are all normalized
    float denom = dot(normal, dir);
    
    if (denom > kEpsilon) {
        Eigen::Vector3f p0l0 = center - eye;
        t = dot(p0l0, normal) / denom;
    }

    if (t < 0) return -1;

    if (radius < 0) {
        return t;
    }

    Eigen::Vector3f p = eye + dir * t;
    Eigen::Vector3f v = p - center;
    float d2 = dot(v, v);

    if (d2 <= radius*radius) return t;
    
    return -1;
}
