#include <Eigen/Core>
#include <math.h>
#include "VectorMath.hpp"
#include "Vector3f.h"
#include "VectorMath2.h"

float magnitude(Eigen::Vector3f V) {
   return sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2]);
}

Eigen::Vector3f normalize(Eigen::Vector3f V) {
   float mag = magnitude(V);
   return Eigen::Vector3f(V[0] / mag, V[1] / mag, V[2] / mag);
}

Eigen::Vector3f cross(Eigen::Vector3f U, Eigen::Vector3f V) {
   float x = U[1] * V[2] - U[2] * V[1];
   float y = U[2] * V[0] - U[0] * V[2];
   float z = U[0] * V[1] - U[1] * V[0];
   return Eigen::Vector3f(x,y,z); 
}

float dot(Eigen::Vector3f U, Eigen::Vector3f V) {
   float ret = U(0)*V(0) + U(1)*V(1) + U(2)*V(2);
   return ret;
}

float angle(Eigen::Vector3f U, Eigen::Vector3f V) {
   return acos(dot(U, V) / (magnitude(U) * magnitude(V)));
}
