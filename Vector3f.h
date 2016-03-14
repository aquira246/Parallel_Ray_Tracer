#pragma  once
#ifndef __VECTOR3F_H__
#define __VECTOR3F_H__

#include <cuda_runtime.h>

class Vector3f
{
   public:
      __device__ __host__
      Vector3f();
      __device__ __host__
      Vector3f(float a, float b, float c);
      __device__ __host__
      ~Vector3f();
      
      __device__ __host__
      Vector3f Add(Vector3f &other);
      __device__ __host__
      Vector3f Subtract(Vector3f &other);
      __device__ __host__
      Vector3f Dot(Vector3f &other);
      __device__ __host__
      Vector3f Cross(Vector3f &other);
      __device__ __host__
      float Magnitude();
      __device__ __host__
      Vector3f Normalize();

      __device__ __host__ inline
      Vector3f operator+ (const Vector3f& other)
      {
         return Vector3f(this->data[0] + other.data[0],
                         this->data[1] + other.data[1],
                         this->data[2] + other.data[2]);
      }

      __device__ __host__ inline
      Vector3f operator- (const Vector3f& other)
      {
         return Vector3f(this->data[0] - other.data[0],
                         this->data[1] - other.data[1],
                         this->data[2] - other.data[2]);
      }

      __device__ __host__ inline
      Vector3f operator-()
      {
         return Vector3f(-this->data[0], -this->data[1], -this->data[2]);
      }

      __device__ __host__ inline
      Vector3f operator* (const float val)
      {
         return Vector3f(this->data[0] * val,
                         this->data[1] * val,
                         this->data[2] * val);
      }

      __device__ __host__ inline
      float& operator[] (int index)
      {
         return (data[index]); // No OOB checking for efficiency, so be careful
      }

   private:
      float data[3];
};

//template <typename T> __device__ __host__
//operator*(T scalar, const Vector3f &obj);

#endif
