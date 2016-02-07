#pragma  once
#ifndef __Sphere__
#define __Sphere__

#include <Eigen/Dense>
#include <cmath>
#include <math.h>
#include <Ray.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define NUM_MATS 7

struct Material
{
	Eigen::Vector3f ambient, diffuse, specular;
	float shine;
}; 

class Sphere
{
	public:
		Sphere();
		Sphere(Eigen::Vector3f c);
		Sphere(float r);
		Sphere(Eigen::Vector3f c, float r);
		~Sphere();
		
		Material mat;
		Eigen::Vector3f center;
		float radius;

		void SetMaterialToMat(Material newMat); 
		void SetMaterialByNum(int colorNum);
		void SetMaterial(std::string colorName);

		float checkHit(Ray ray);
		float checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir);


	private:
};

//The first part of the vector is how many answers there are, the second part is the first answer, and the third is the second answer
Eigen::Vector3f QuadraticFormula(double A, double B, double C);

#endif
