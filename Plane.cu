#include "Plane.hpp"

#define kEpsilon 1e-8 

Plane::Plane() {
    //SetMaterialByNum(rand() % NUM_MATS);
    center = Vector3f(0,0,0);
    normal = Vector3f(0,0,-1);
    radius = 1.0f;
}

Plane::Plane(Vector3f c, Vector3f n, float r) {
    //SetMaterialByNum(rand() % NUM_MATS);
    center = c;
    normal = n;
    radius = r;
}

Plane::~Plane(){

}

__device__ __host__
Vector3f Plane::GetNormal(Vector3f hitPt) {
   return normal;
}

__device__ __host__
float Plane::checkHit(Vector3f eye, Vector3f dir) {
    
    float t = -1;

    // assuming vectors are all normalized
    float denom = dot(normal, dir);

    if (fabs(denom) > kEpsilon) {
        Vector3f p0l0 = center - eye;
        t = dot(p0l0, normal) / denom;
    }

    if (t < 0) return -1;

    if (radius < 0) {
        return t;
    }

    Vector3f p = eye + dir * t;
    Vector3f v = p - center;
    float d2 = dot(v, v);

    if (d2 <= radius*radius) return t;
    
    return -1;
}
