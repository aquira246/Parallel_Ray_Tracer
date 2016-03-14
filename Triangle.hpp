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

      virtual Vector3f GetNormal(Vector3f hitPt);
      float checkHit(Vector3f eye, Vector3f dir);

   protected:
   // Parts of a triangle
   Vector3f a, b, c;
   Vector3f normal;
   float areaSqr;

   void Initialize();
};

#endif
