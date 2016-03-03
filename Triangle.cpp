#include <Triangle.hpp>

using namespace std;

constexpr float kEpsilon = 1e-8; 

Triangle::Triangle() {
    SetMaterialByNum(rand() % NUM_MATS);
    v0 = Eigen::Vector3f();
    v1 = Eigen::Vector3f();
    v2 = Eigen::Vector3f();
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
    areaSqr = normal.norm();
}

// TODO! RETURN UV SO YOU CAN INTERPOLATE COLORS
//  u * cols[0] + v * cols[1] + (1 - u - v) * cols[2];
// http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
float Triangle::checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir) {
    Eigen::Vector3f v0v1 = v1 - v0;
    Eigen::Vector3f v0v2 = v2 - v0;
    Eigen::Vector3f pvec = dir.cross(v0v2);
    float det = v0v1.dot(pvec);

    #ifdef CULLING
    // if the determinant is negative the triangle is backfacing and can be culled
    // if the determinant is close to 0, the ray misses the triangle
    if (det < kEpsilon) return 0;
    #else
    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < kEpsilon) return 0;
    #endif

    float invDet = 1 / det;

    Eigen::Vector3f tvec = eye - v0;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0 || u > 1) return 0;

    Eigen::Vector3f qvec = tvec.cross(v0v1);
    float v = dir.dot(qvec) * invDet;
    if (v < 0 || u + v > 1) return 0;

    float t = v0v2.dot(qvec) * invDet;

    return t;
} 