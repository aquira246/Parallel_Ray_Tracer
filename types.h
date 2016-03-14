#ifndef __TYPES_H__
#define __TYPES_H__

#include <vector>
#include "Triangle.hpp"
#include "Sphere.hpp"
#include "Vector3f.h"

/* Color struct */
typedef struct color_struct {
   double r;
   double g;
   double b;
   double f; // "filter" or "alpha"
} color_t;

typedef struct Camera {
   Vector3f position;
   Vector3f look_at;
   Vector3f right;
   Vector3f up;
} Camera;

typedef struct Light {
   color_t color;
   Vector3f location;
} Light;

typedef struct DirLight {
   color_t color;
   Vector3f direction;
} DirLight;

#endif
