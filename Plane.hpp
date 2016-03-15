#pragma  once
#ifndef __PLANE_H__
#define __PLANE_H__

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include "Shape.hpp"
#include "Ray.hpp"
#include "Vector3f.h"
#include "VectorMath.h"

class Plane: public Shape
{
   public:
      Plane();
      Plane(Vector3f c, Vector3f n, float r);
      ~Plane();

      Vector3f normal;
        
      __device__ __host__
      virtual Vector3f GetNormal(Vector3f hitPt);
      // Shape has a center and radius, the only components of a Plane
      __device__ __host__
      float checkHit(Vector3f eye, Vector3f dir);

   private:
};

#endif
