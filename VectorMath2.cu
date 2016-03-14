#include <math.h>
#include "VectorMath2.h"
#include "Vector3f.h"

__device__ __host__
float magnitude(Vector3f V) {
   return sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2]);
}

__device__ __host__
Vector3f normalize(Vector3f V) {
   float mag = magnitude(V);
   return Vector3f(V[0] / mag, V[1] / mag, V[2] / mag);
}

__device__ __host__
Vector3f cross(Vector3f U, Vector3f V) {
   float x = U[1] * V[2] - U[2] * V[1];
   float y = U[2] * V[0] - U[0] * V[2];
   float z = U[0] * V[1] - U[1] * V[0];
   return Vector3f(x,y,z); 
}

__device__ __host__
float dot(Vector3f U, Vector3f V) {
   float ret = U[0]*V[0] + U[1]*V[1] + U[2]*V[2];
   return ret;
}

__device__ __host__
float angle(Vector3f U, Vector3f V) {
   return acos(dot(U, V) / (magnitude(U) * magnitude(V)));
}
