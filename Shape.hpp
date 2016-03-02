#pragma  once
#ifndef __Shape__
#define __Shape__

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

        void SetMaterialToMat(Material newMat); 
        void SetMaterialByNum(int colorNum);
        void SetMaterial(std::string colorName);

        float checkHit(Ray ray) {
            return checkHit(ray.eye, ray.direction);
        }

        float checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir) {
            return 0;
        }


    private:
};

#endif