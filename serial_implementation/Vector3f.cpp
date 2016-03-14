#include "Vector3f"

Vector3f::Vector3f() {
   
}

Vector3f::Vector3f(float a, float b, float c) {
   data[0] = a;
   data[1] = b;
   data[2] = c;
}

Vector3f::~Vector3f() {

}
      
Vector3f Vector3f::Add(Vector3f other) {
   return this + other;
}

Vector3f Vector3f::Subtract(Vector3f other) {

}

Vector3f Vector3f::Dot(Vector3f other) {

}

Vector3f Vector3f::Cross(Vector3f other) {

}

Vector3f Vector3f::Magnitude() {

}

Vector3f Vector3f::Normalize() {

}
