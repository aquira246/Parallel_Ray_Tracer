#pragma once
#ifndef __VECTOR_MATH_H__
#define __VECTOR_MATH_H__

#include <cuda_runtime.h>
#include "Vector3f.h"

__device__ __host__
float magnitude(Vector3f V);
__device__ __host__
Vector3f normalize(Vector3f V);
__device__ __host__
Vector3f cross(Vector3f U, Vector3f V);
__device__ __host__
float dot(Vector3f U, Vector3f V);
__device__ __host__
float angle(Vector3f U, Vector3f V);

#endif

