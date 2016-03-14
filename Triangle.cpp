#include "Triangle.hpp"

using namespace std;

constexpr float kEpsilon = 1e-8; 

Triangle::Triangle() {
   //SetMaterialByNum(rand() % NUM_MATS);
   a = Vector3f();
   b = Vector3f();
   c = Vector3f();
   Initialize();
}

Triangle::Triangle(Vector3f pta, Vector3f ptb, Vector3f ptc) {
   //SetMaterialByNum(rand() % NUM_MATS);
   a = pta;
   b = ptb;
   c = ptc;
   Initialize();
}

Triangle::~Triangle(){

}

void Triangle::Initialize() {
   Vector3f ab = b - a;
   Vector3f ac = c - a;

   // compute plane's normal
   normal = normalize(cross(ab, ac));
   //areaSqr = magnitude(normal);
   // the not offsetted center of the circumsphere
   center = cross(normal, ab) * magnitude(ac) + cross(ac, normal) * magnitude(ab);
   // radius ofthe circumsphere
   radius = magnitude(center);
   // offset the center properly in the world
   center = center + a;
}

Vector3f Triangle::GetNormal(Vector3f hitPt) {
   return normal;
}

// http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
float Triangle::checkHit(Vector3f eye, Vector3f dir) {
   float u, v, t;

   // first check for circumsphere hit
   Vector3f dist = eye - center;

   float A = dot(dir, dir);
   float B = dot((dir * 2), dist);
   float C = dot(dist, dist) - radius*radius;

   Vector3f quad = QuadraticFormula(A, B, C);
   float result;

   if (quad[0] == 0) {
      //SHOULD BE AN ERROR
      result = 0;
   }

   if (quad[0] == 1) {
      result = quad[1];
   }

   if (fabs(quad[1]) <= fabs(quad[2])) {
      result = quad[1];
   } else {
      result = quad[2];
   }

   // failure to even hit the circumsphere
   if (result < 0) {
      return 0;
   }

   Vector3f ab = (b - a);
   Vector3f ac = (c - a);
   Vector3f pvec = cross(dir, ac);
   float det = dot(ab, pvec);
   #ifdef CULLING
   // if the determinant is negative the triangle is backfacing
   // if the determinant is close to 0, the ray misses the triangle
   if (det < kEpsilon) return 0;
   #else
   // ray and triangle are parallel if det is close to 0
   if (fabs(det) < kEpsilon) return 0;
   #endif
   float invDet = 1 / det;

   Vector3f tvec = eye - a;
   u = dot(tvec, pvec) * invDet;
   if (u < 0 || u > 1) return 0;

   Vector3f qvec = normalize(cross(tvec, ab));
   v = dot(dir, qvec) * invDet;
   if (v < 0 || u + v > 1) return 0;

   t = dot(ac, qvec) * invDet;

   return t;
}
