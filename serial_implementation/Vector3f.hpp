#pragma  once
#ifndef __VECTOR3F_H__
#define __VECTOR3F_H__

class Vector3f
{
   public:
      Vector3f();
      Vector3f(float a, float b, float c);
      ~Vector3f();
      
      Vector3f Add(Vector3f other);
      Vector3f Subtract(Vector3f other);
      Vector3f Dot(Vector3f other);  
      Vector3f Cross(Vector3f other);
      Vector3f Magnitude();
      Vector3f Normalize();

      Vector3f operator+ (const Vector3f& other)
      {
         return Vector3f(this->data[0] + other.data[0],
                         this->data[1] + other.data[1],
                         this->data[2] + other.data[2]);
      }

      Vector3f operator- (const Vector3f& other);
      {
         return Vector3f(this->data[0] - other.data[0],
                         this->data[1] - other.data[1],
                         this->data[2] - other.data[2]);
      }

      int& operator[] (int index);
      {
         return data[index]; // No OOB checking for efficiency, so be careful
      }

   private:
      float data[3];
};

#endif
