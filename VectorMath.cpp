#include <Eigen/Dense>
#include "VectorMath.hpp"

float magnitude(Eigen::Vector3f V) {
   return sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2]);
}

float magnitude(Vec3 V) {
   return sqrt(V.x * V.x + V.y * V.y + V.z * V.z);
}

Eigen::Vector3f normalize(Eigen::Vector3f V) {
   float mag = magnitude(V);
   return Eigen::Vector3f(V[0] / mag, V[1] / mag, V[2] / mag);
}

Vec3 normalize(Vec3 V) {
   float mag = magnitude(V);
   Vec3 ret;
   ret.x = V.x / mag;
   ret.y = V.y / mag;
   ret.z = V.z / mag;
   return ret;
}

Eigen::Vector3f cross(Eigen::Vector3f U, Eigen::Vector3f V) {
   float x = U[1] * V[2] - U[2] * V[1];
   float y = U[2] * V[0] - U[0] * V[2];
   float z = U[0] * V[1] - U[1] * V[0];
   return normalize(Eigen::Vector3f(x,y,z)); 
}

Vec3 cross(Vec3 U, Vec3 V) {
   Vec3 ret;
   ret.x = U.y * V.z - U.z * V.y;
   ret.y = U.z * V.x - U.x * V.z;
   ret.z = U.x * V.y - U.y * V.x;
   return normalize(ret); 
}

float dot(Eigen::Vector3f U, Eigen::Vector3f V) {
   float ret = U[0]*V[0] + U[1]*V[1] + U[2]*V[2];
   return ret;
}

float dot(Vec3 U, Vec3 V) {
   float ret = U.x*V.x + U.y*V.y + U.z*V.z;
   return ret;
}
