#pragma  once
#ifndef __SHAPE__
#define __SHAPE__

#include <Shape.hpp>
#include <Eigen/Dense>
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

class Shape
{
    public:
        Shape();
        ~Shape();
        
        Material mat;
        Eigen::Vector3f center;

        void SetMaterialToMat(Material newMat); 
        void SetMaterialByNum(int colorNum);
        void SetMaterial(std::string colorName);

        virtual float checkHit(Ray ray) {
            return checkHit(ray.eye, ray.direction);
        }

        virtual float checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir) {
            //std::cout << "BAD BAD BAD!" << std::endl;
            return 0;
        }


    private:
};

// return vector: inxex 1: how many answers there are
// index 2: the positive output
// index 3: the negative output
Eigen::Vector3f QuadraticFormula(double A, double B, double C);

#endif