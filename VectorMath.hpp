#ifndef __VECTOR_MATH_H__
#define __VECTOR_MATH_H__

#include <Eigen/Dense>

typedef struct Vec3
{
   float x, y, z;
}Vec3;

float magnitude(Eigen::Vector3f V);
float magnitude(Vec3 V);

Eigen::Vector3f normalize(Eigen::Vector3f V);
Vec3 normalize(Vec3 V);

Eigen::Vector3f cross(Eigen::Vector3f U, Eigen::Vector3f V);
Vec3 cross(Vec3 U, Vec3 V);

float dot(Eigen::Vector3f U, Eigen::Vector3f V);
float dot(Vec3 U, Vec3 V);

#endif
