#include "Plane.hpp"

using namespace std;

#define kEpsilon 1e-5 

Plane::Plane() {
   center = Eigen::Vector3f(0,0,0);
   normal = Eigen::Vector3f(0,0,-1);
   radius = 1.0f;
   #ifndef CULLING
   isFlat = true;
   #endif
}

Plane::Plane(Eigen::Vector3f c, Eigen::Vector3f n, float r) {
   center = c;
   normal = n;
   radius = r;
   #ifndef CULLING
   isFlat = true;
   #endif
}

Plane::~Plane(){

}

Eigen::Vector3f Plane::GetNormal(Eigen::Vector3f hitPt) {
   return normal;
}

float Plane::checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir) {
    
    float t = -1;

    // assuming vectors are all normalized
    double denom = dot(normal, dir);

    if (fabs(denom) > kEpsilon) {
        Eigen::Vector3f p0l0 = center - eye;
        t = dot(p0l0, normal) / denom;
    }

    if (t < 0) return 0;

    if (radius < 0) {
        return t;
    }

    Eigen::Vector3f p = eye + dir * t;
    Eigen::Vector3f v = p - center;
    double d2 = dot(v, v);

    if (d2 <= radius*radius) return t;
    
    return 0;
}
