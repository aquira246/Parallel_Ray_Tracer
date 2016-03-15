#include "Triangle.hpp"

using namespace std;

constexpr float kEpsilon = 1e-8; 

Triangle::Triangle() {
   //SetMaterialByNum(rand() % NUM_MATS);
   a = Eigen::Vector3f();
   b = Eigen::Vector3f();
   c = Eigen::Vector3f();
   Initialize();
}

Triangle::Triangle(Eigen::Vector3f pta, Eigen::Vector3f ptb, Eigen::Vector3f ptc) {
   //SetMaterialByNum(rand() % NUM_MATS);
   a = pta;
   b = ptb;
   c = ptc;
   Initialize();
}

Triangle::~Triangle(){

}

void Triangle::Initialize() {
   // compute plane's normal
   Eigen::Vector3f ab = b - a;
   Eigen::Vector3f ac = c - a;

   // no need to normalize
   normal = cross(ab, ac);
   areaSqr = normal.norm();
   // the not offsetted center of the circumsphere
   center = cross(normal, ab) * magnitude(ac) + cross(ac, normal) * magnitude(ab);
   // radius ofthe circumsphere
   radius = magnitude(center);
   // offset the center properly in the world
   center += a;
}

Eigen::Vector3f Triangle::GetNormal(Eigen::Vector3f hitPt) {
   return normal;
}

// TODO! RETURN UV SO YOU CAN INTERPOLATE COLORS
//  u * cols[0] + v * cols[1] + (1 - u - v) * cols[2];
// http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
float Triangle::checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir) {
   double u, v, t;

   // first check for circumsphere hit
   Eigen::Vector3f dist = eye - center;

   double A = dot(dir, dir);
   double B = dot((2*dir), dist);
   double C = dot(dist, dist) - radius*radius;

   Eigen::Vector3f quad = QuadraticFormula(A, B, C);
   float result;

   if (quad(0) == 0) {
      //SHOULD BE AN ERROR
      result = 0;
   }

   if (quad(0) == 1) {
      result = quad(1);
   }

   if (fabs(quad(1)) <= fabs(quad(2))) {
      result = quad(1);
   } else {
      result = quad(2);
   }

   // failure to even hit the circumsphere
   if (result < 0) {
      return 0;
   }

   Eigen::Vector3f ab = b - a;
   Eigen::Vector3f ac = c - a;
   Eigen::Vector3f pvec = cross(dir, ac);
   float det = dot(ab, pvec);
   #ifdef CULLING
   // if the determinant is negative the triangle is backfacing
   // if the determinant is close to 0, the ray misses the triangle
   if (fabs(det) < kEpsilon) return 0;
   #else
   // ray and triangle are parallel if det is close to 0
   if (det < kEpsilon) return 0;
   #endif
   float invDet = 1 / det;

   Eigen::Vector3f tvec = eye - a;
   u = dot(tvec, pvec) * invDet;
   if (u < 0 || u > 1) return 0;

   Eigen::Vector3f qvec = cross(tvec, ab);
   v = dot(dir, qvec) * invDet;
   if (v < 0 || u + v > 1) return 0;

   t = dot(ac, qvec) * invDet;

   return t;
}
