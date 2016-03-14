#include "Vector3f.h"
#include <math.h>

__device__ __host__
Vector3f::Vector3f() {
   data[0] = 0.0f;
   data[1] = 0.0f;
   data[2] = 0.0f;
}

__device__ __host__
Vector3f::Vector3f(float a, float b, float c) {
   data[0] = a;
   data[1] = b;
   data[2] = c;
}

__device__ __host__
Vector3f::~Vector3f() {

}

__device__ __host__      
Vector3f Vector3f::Add(Vector3f &other) {
   return Vector3f(data[0] + other.data[0],
                   data[1] + other.data[1],
                   data[2] + other.data[2]);
}

__device__ __host__
Vector3f Vector3f::Subtract(Vector3f &other) {
   return Vector3f(this->data[0] - other.data[0],
                   this->data[1] - other.data[1],
                   this->data[2] - other.data[2]);
}

__device__ __host__
Vector3f Vector3f::Dot(Vector3f &other) {
   return Vector3f(this->data[0] * this->data[0], this->data[1] * other.data[1], other.data[2] * other.data[2]);
}

__device__ __host__
Vector3f Vector3f::Cross(Vector3f &other) {
   return Vector3f(this->data[1] * other.data[2] - this->data[2] * other.data[1],
                   this->data[2] * other.data[0] - this->data[0] * other.data[2],
                   this->data[0] * other.data[1] - this->data[1] * other.data[0]);
}

__device__ __host__
float Vector3f::Magnitude() {
   return sqrt(this->data[0] * this->data[0] + this->data[1] * this->data[1] + this->data[2] * this->data[2]);
}

__device__ __host__
Vector3f Vector3f::Normalize() {
   float mag = this->Magnitude();
   return Vector3f(this->data[0] / mag, this->data[1] / mag, this->data[2] / mag);
}
/*
template <typename T, Vector3f &obj> __device__ __host__
operator*(T scalar, const Vector3f &obj)
{
   return Vector3f(obj.data[0] * scalar,
                   obj.data[1] * scalar,
                   obj.data[2] * scalar);
}
*/
