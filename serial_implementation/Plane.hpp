#pragma  once
#ifndef __Plane__
#define __Plane__

#include <Eigen/Dense>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include "Shape.hpp"
#include "Ray.hpp"
#include "VectorMath.hpp"

class Plane: public Shape
{
    public:
        Plane();
        Plane(Eigen::Vector3f c, Eigen::Vector3f n, float r);
        ~Plane();

        Eigen::Vector3f normal;
        
        virtual Eigen::Vector3f GetNormal(Eigen::Vector3f hitPt);
        // Shape has a center and radius, the only components of a Plane
        float checkHit(Eigen::Vector3f eye, Eigen::Vector3f dir);

    private:
};

#endif
