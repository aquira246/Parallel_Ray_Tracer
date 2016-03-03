#pragma  once
#ifndef __Triangle__
#define __Triangle__

#include <Shape.hpp>
#include <Eigen/Dense>
#include <math.h>
#include <Ray.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 

class Triangle: public Shape
{
    public:
        Triangle();
        Triangle(Eigen::Vector3f pta, Eigen::Vector3f ptb, Eigen::Vector3f ptc);
        ~Triangle();

        float checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir);

    protected:
    // Parts of a triangle
    Eigen::Vector3f v0, v1, v2;
    Eigen::Vector3f normal;
    float areaSqr;
    
    void Initialize();
};

#endif
