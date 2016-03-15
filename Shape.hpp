#pragma  once
#ifndef __SHAPE_H__
#define __SHAPE_H__

#include <math.h>
#include <iostream>
#include "Vector3f.h"
#include "VectorMath.h"
#include "Ray.hpp"

#define NUM_MATS 7


// POV-ray material
struct Material
{
   Vector3f rgb;
   float ambient, diffuse, specular, roughness, shine;
} typedef Material;


class Shape
{
   public:
      Shape();
      ~Shape();
        
      Material mat;
      // No matter the shape we generate a bounding sphere
      Vector3f center;
      float radius;

      void SetMaterialToMat(Material newMat); 
      void SetMaterialByNum(int colorNum);
      void SetMaterial(std::string colorName);

      __device__ __host__
      virtual Vector3f GetNormal(Vector3f hitPt) {
         return Vector3f(); 
      }

      __device__ __host__
      float checkHit(Ray ray) {
         return checkHit(ray.eye, ray.direction);
      }

      __device__ __host__
      virtual float checkHit(Vector3f eye, Vector3f dir) {
         return 0;
      }


   private:
};

// return vector: inxex 1: how many answers there are
// index 2: the positive output
// index 3: the negative output
__device__ __host__
Vector3f QuadraticFormula(double A, double B, double C);

#endif
