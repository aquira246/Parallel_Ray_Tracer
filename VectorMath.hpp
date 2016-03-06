#ifndef __VECTOR_MATH_H__
#define __VECTOR_MATH_H__

#include <Eigen/Dense>

float magnitude(Eigen::Vector3f V);
Eigen::Vector3f normalize(Eigen::Vector3f V);
Eigen::Vector3f cross(Eigen::Vector3f U, Eigen::Vector3f V);

#endif
