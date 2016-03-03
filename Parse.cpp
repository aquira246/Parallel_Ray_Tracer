/*
 * Base code for this parser from http://mrl.nyu.edu/~dzorin/cg05/handouts/pov-parser/index.html
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <math.h> 
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "Tokens.hpp"

using namespace Eigen;
using namespace std;

/* Functions in this file implement parsing operations for 
each object type. The main function to be called is Parse().
Each parsing function assigns values retrieved from the file 
to a set of local variables that can be used to create 
objects. Parsing stops at the first error and the program 
exits through function Error() defined in tokens.c, after
printing the current line number and an error message. 
(A real parser would have better error recovery, but this 
simple approach greatly simplifies the code and is sufficient
for our purposes).

Although this code runs, it requires additions to make it fully 
functional. Search for comments starting with TODO to find 
places requiring modification.
*/

struct Finish { 
   double ambient;
   double diffuse;
   double specular;
   double roughness;
   double phong;
   double phong_size;
   double reflection;
   int metallic;
} typedef Finish;

struct ModifierStruct { 
   Vector4f pigment;
   struct Finish finish;
   double interior;
} typedef ModifierStruct;

/* a collection of functions for syntax verification */
void ParseLeftAngle() { 
  GetToken();
  if(Token.id != T_LEFT_ANGLE ) Error("Expected <");
}

void ParseRightAngle() { 
  GetToken();
  if(Token.id != T_RIGHT_ANGLE ) Error("Expected >");  
}

double ParseDouble() { 
  GetToken();
  if(Token.id != T_DOUBLE ) Error("Expected a number");
  return Token.double_value;
}

void ParseLeftCurly() { 
  GetToken();
  if(Token.id != T_LEFT_CURLY ) Error("Expected {");
}

void ParseRightCurly() { 
  GetToken();
  if(Token.id != T_RIGHT_CURLY ) Error("Expected }");
}

void ParseComma() { 
  GetToken();
  if(Token.id != T_COMMA ) Error("Expected ,");
}

void ParseVector(Vector3f &v) { 
  ParseLeftAngle();
  v[0] = ParseDouble();
  ParseComma();
  v[1] = ParseDouble();
  ParseComma();
  v[2] = ParseDouble();
  ParseRightAngle();
}

void ParseRGBFColor(Vector4f &c) {
  ParseLeftAngle();
  c[0] = ParseDouble();
  ParseComma();
  c[1] = ParseDouble();
  ParseComma();
  c[2] = ParseDouble();
  ParseComma();
  c[3] = ParseDouble();
  ParseRightAngle();
}

void ParseRGBColor(Vector4f &c) { 
  ParseLeftAngle();
  c[0] = ParseDouble();
  ParseComma();
  c[1] = ParseDouble();
  ParseComma();
  c[2] = ParseDouble();
  c[3] = 0.0;
  ParseRightAngle();
}

void ParseColor(Vector4f &c) { 
  GetToken();
  if(Token.id == T_RGB) 
    ParseRGBColor(c);
  else if ( Token.id == T_RGBF )
    ParseRGBFColor(c);
  else Error("Expected rgb or rgbf");
}

void PrintColor(Vector4f &c) { 
  printf("rgbf <%.3g,%.3g,%.3g,%.3g>", c[0], c[1], c[2], c[3]);
}

void ParsePigment(Vector4f &pigment) { 
  ParseLeftCurly();
  while(1) { 
    GetToken();
    if(Token.id == T_COLOR)
      ParseColor(pigment);
    else if(Token.id == T_RIGHT_CURLY) return;
    else Error("error parsing pigment: unexpected token");
  }    
}

void PrintPigment(Vector4f &pigment) {
  printf("\tpigment { color ");
  PrintColor(pigment);
  printf("}");
}

void ParseFinish(Finish &finish) { 
  ParseLeftCurly();
  while(1) { 
    GetToken();
    switch(Token.id) {
    case T_AMBIENT:
      finish.ambient = ParseDouble();
      break;
    case T_DIFFUSE:
      finish.diffuse = ParseDouble();
      break;
    case T_SPECULAR:
      finish.specular= ParseDouble();
      break;
    case T_ROUGHNESS:
      finish.roughness= ParseDouble();
      break;
    case T_PHONG:
      finish.phong = ParseDouble();
      break;
    case T_PHONG_SIZE:
      finish.phong_size = ParseDouble();
      break;
    case T_REFLECTION:
      finish.reflection = ParseDouble();
      break;
    case T_METALLIC:
      finish.metallic = (ParseDouble() != 0.0 ? 1: 0);
      break;
    case T_RIGHT_CURLY: return;
    default: Error("Error parsing finish: unexpected token");
    }   
  } 
}

void PrintFinish(struct Finish &finish) { 
  printf("\tfinish { ambient %.3g diffuse %.3g phong %.3g phong_size %.3g reflection %.3g metallic %d }\n", 
         finish.ambient, finish.diffuse, 
         finish.phong, finish.phong_size, 
         finish.reflection, finish.metallic);
}

void ParseInterior(double interior) { 
  ParseLeftCurly();
  while(1) {
    GetToken();
    if( Token.id == T_RIGHT_CURLY) return;
    else if (Token.id == T_IOR) { 
      interior = ParseDouble();
    } else Error("Error parsing interior: unexpected token\n");
  }
}


void InitModifiers(struct ModifierStruct &modifiers) { 
  modifiers.pigment[0] = 0;
  modifiers.pigment[1] = 0;
  modifiers.pigment[2] = 0;
  modifiers.pigment[3] = 0;

  modifiers.finish.ambient    = 0.1;
  modifiers.finish.diffuse    = 0.6;
  modifiers.finish.phong      = 0.0;
  modifiers.finish.phong_size = 40.0;
  modifiers.finish.reflection = 0;

  modifiers.interior = 1.0; 
}


void ParseModifiers(ModifierStruct &modifiers) { 
  while(1) { 
    GetToken();
    switch(Token.id) { 
    case T_SCALE:
    case T_ROTATE:
    case T_TRANSLATE:
    case T_PIGMENT:
      ParsePigment(modifiers.pigment);
      break;
    case T_FINISH:
      ParseFinish(modifiers.finish);
      break;
    case T_INTERIOR:
      ParseInterior(modifiers.interior);
      break;      
    default: UngetToken(); return;
    }
  }
}

void PrintModifiers(ModifierStruct &modifiers) {
  PrintPigment(modifiers.pigment);
  printf("\n"); 
  PrintFinish(modifiers.finish);
  printf("\tinterior { ior %.3g }\n", modifiers.interior);
}


void ParseCamera() { 
  Vector3f location, right, up, look_at; 
  double angle;
  int done = FALSE;

  /* default values */
  location = Vector3f(0.0, 0.0, 0.0);   
  look_at = Vector3f(0.0, 0.0, 1.0);
  right = Vector3f(1.0, 0.0, 0.0);
  up = Vector3f(0.0, 1.0, 0.0);
  angle = 60.0 * M_PI / 180.0;

  ParseLeftCurly();

  /* parse camera parameters */  
  while(!done) {     
    GetToken();
    switch(Token.id) { 
    case T_LOCATION:
      ParseVector(location);
      break;
    case T_RIGHT:
      ParseVector(right);
      break;
    case T_UP:
      ParseVector(up);
      break;
    case T_LOOK_AT:
      ParseVector(look_at);
      break;
    case T_ANGLE:
      angle = M_PI * ParseDouble() / 180.0;
      break;
    default:
      done = TRUE;
      UngetToken();
      break;
    }
  }

  ParseRightCurly();
  //TODO?
}

void ParseSphere() { 
   Vector3f center; 
   double radius; 
   ModifierStruct modifiers;

   InitModifiers(modifiers);
   center = Vector3f(0, 0, 0);
   radius = 1.0;

   ParseLeftCurly();
   ParseVector(center);
   ParseComma();
   radius = ParseDouble();

   ParseModifiers(modifiers);
   ParseRightCurly();
}

void ParseTriangle() { 
   Vector3f vert1, vert2, vert3;
   ModifierStruct modifiers;
   InitModifiers(modifiers);

   ParseLeftCurly();

   ParseVector(vert1);
   ParseComma();
   ParseVector(vert2);
   ParseComma();
   ParseVector(vert3);
   ParseModifiers(modifiers);

   ParseRightCurly();
}

void ParsePlane() { 
   Vector3f plane;
   float radius;
   ModifierStruct modifiers;
   InitModifiers(modifiers);

   ParseLeftCurly();

   ParseVector(plane);
   ParseComma();
   radius = (float) ParseDouble();
   ParseModifiers(modifiers);

   ParseRightCurly();
}

void ParseLightSource() { 
   Vector4f c = Vector4f(0, 0, 0, 0);
   Vector3f pos = Vector3f(0, 0, 0);
   ParseLeftCurly();
   ParseVector(pos);
   GetToken();
   if(Token.id != T_COLOR) Error("Error parsing light source: missing color");
   ParseColor(c);
   ParseRightCurly();
} 

void ParseGlobalSettings() { 
   Vector4f ambient = Vector4f(0, 0, 0, 0);
   ParseLeftCurly();
   while(1) { 
     GetToken();
     if(Token.id == T_AMBIENT_LIGHT) {
       ParseLeftCurly();
       GetToken();
       if(Token.id != T_COLOR) 
         Error("Error parsing light source: missing color");
       ParseColor(ambient);
       ParseRightCurly();
     } else if(Token.id == T_RIGHT_CURLY) { 
       break;
     } else         
       Error("error parsing default settings: unexpected token");
   }
}

/* main parsing function calling functions to parse each object;  */
int Parse(FILE* infile) {
   int numObjects = 0;
   InitializeToken(infile);
   GetToken();
   while(Token.id != T_EOF) { 
      switch(Token.id) { 
         case T_CAMERA:
            ParseCamera();
            break;
         case T_TRIANGLE:
            ParseTriangle();
            break;
         case T_SPHERE:
            ParseSphere();
            break;
         case T_PLANE:
            ParsePlane();
            break;
         case T_LIGHT_SOURCE:
            ParseLightSource();
            break;
         case T_GLOBAL_SETTINGS:
            ParseGlobalSettings();
            break;
         default: Error("Unknown statement");
      }
      GetToken();
      ++numObjects;
   }
   return numObjects;
}
