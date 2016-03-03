#include <Triangle.hpp>

using namespace std;

Triangle::Triangle() {
    SetMaterialByNum(rand() % NUM_MATS);
    v0 = 0;
    v1 = 0;
    v2 = 0;
    Initialize();
}

Triangle::Triangle(Eigen::Vector3f pt1, Eigen::Vector3f pt2, Eigen::Vector3f pt3) {
    SetMaterialByNum(rand() % NUM_MATS);
    v0 = pt1;
    v1 = pt2;
    v2 = pt3;
    Initialize();
}

Triangle::~Triangle(){

}

void Triangle::Initialize() {
    // compute plane's normal
    Eigen::Vector3f v0v1 = v1 - v0;
    Eigen::Vector3f v0v2 = v2 - v0;

    // no need to normalize
    normal = v0v1.cross(v0v2);
    areaSqr = normal.length();
}

// http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
float Triangle::checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir) {
    float t;

    // Step 1: finding P
    // check if ray and plane are parallel ?
    float NdotRayDirection = normal.dot(dir);
    
    if (fabs(NdotRayDirection) < kEpsilon) // almost 0
       return 0; // they are parallel so they don't intersect !

    // compute d parameter using equation 2
    float d = normal.dotProduct(v0);

    // compute t (equation 3)
    t = (normal.dotProduct(eye) + d) / NdotRayDirection;
    // check if the triangle is in behind the ray
    if (t < 0) return 0; // the triangle is behind

    // compute the intersection point using equation 1
    Vec3f P = eye + t * dir;

    // Step 2: inside-outside test
    Vec3f C; // vector perpendicular to triangle's plane

    // edge 0
    Vec3f edge0 = v1 - v0;
    Vec3f vp0 = P - v0;
    C = edge0.crossProduct(vp0);
    if (normal.dotProduct(C) < 0) return 0; // P is on the right side

    // edge 1
    Vec3f edge1 = v2 - v1;
    Vec3f vp1 = P - v1;
    C = edge1.crossProduct(vp1);
    if (normal.dotProduct(C) < 0) return 0; // P is on the right side

    // edge 2
    Vec3f edge2 = v0 - v2;
    Vec3f vp2 = P - v2;
    C = edge2.crossProduct(vp2);
    if (normal.dotProduct(C) < 0) return 0; // P is on the right side;

    return t; // this ray hits the triangle at t
} 