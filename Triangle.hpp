#pragma  once
#ifndef __Triangle__
#define __Triangle__

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include "Ray.hpp"
#include "VectorMath.h"
#include "Shape.hpp"

class Triangle: public Shape
{
   public:
      Triangle();
      Triangle(Vector3f pta, Vector3f ptb, Vector3f ptc);
      ~Triangle();

      __device__ __host__
      virtual Vector3f GetNormal(Vector3f hitPt);
      __device__ __host__
      float checkHit(Vector3f eye, Vector3f dir);

      // Parts of a triangle
      Vector3f a, b, c;
      Vector3f normal;
      float areaSqr;

   void Initialize();
};

#endif
