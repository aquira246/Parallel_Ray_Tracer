#include "Shape.hpp"

using namespace std;

Shape::Shape() {
    //SetMaterialByNum(rand() % NUM_MATS);
    //center = Eigen::Vector3f(0,0,0);
}

Shape::~Shape(){

}

void Shape::SetMaterialToMat(Material newMat) {
   mat = newMat;
}

void Shape::SetMaterialByNum(int colorNum) {
    Eigen::Vector3f col;
    float shine, a, s, d, r;

    switch (colorNum) {
        case 0:
            col = Eigen::Vector3f(.8, .2, .2);
            a = .1;
            d = .7;
            s = .5;
            r = .3;
            shine = 16;
            break;
        
        case 1:
            col = Eigen::Vector3f(.2, .2, .8);
            a = .1;
            d = .7;
            s = .5;
            r = .4;
            shine = 128;
            break;

        case 2:
            col = Eigen::Vector3f(.2, .8, .2);
            a = .1;
            d = .7;
            s = .5;
            r = .6;
            shine = 8;
            break;

        case 3:
            col = Eigen::Vector3f(.8, .2, .8);
            a = .1;
            d = .7;
            s = .5;
            r = .6;
            shine = 128;
            break;

        case 4:
            col = Eigen::Vector3f(.8, .2, .8);
            a = .1;
            d = .7;
            s = .5;
            r = .6;
            shine = 16;
            break;

        case 5:
            col = Eigen::Vector3f(.8, .8, .0);
            a = .1;
            d = .7;
            s = .5;
            r = .7;
            shine = 128;
            break;

        default:
            col = Eigen::Vector3f(.6, .6, .6);
            a = .1;
            d = .5;
            s = .9;
            r = .8;
            shine = 256;
            break;
    }

    mat.ambient = a;
    mat.diffuse = d;
    mat.specular = s;
    mat.shine = shine;
    mat.rgb = col;
    mat.roughness = r;
}

void Shape::SetMaterial(string colorName) {
    if(colorName == "red") {
        SetMaterialByNum(0);
    }
    
    else if(colorName == "blue") {
        SetMaterialByNum(1);
    }

    else if(colorName == "green"){
        SetMaterialByNum(2);
    }

    else if(colorName == "purple"){
        SetMaterialByNum(3);
    }

    else if(colorName == "teal"){
        SetMaterialByNum(4);
    }

    else if(colorName == "orange"){
        SetMaterialByNum(5);
    }

    else {
        cout << "ERROR! " << colorName << " is not a valid color! Here is teal, the color you should have picked" << endl;
        SetMaterialByNum(4);
    }
}

// return vector: inxex 1: how many answers there are
// index 2: the positive output
// index 3: the negative output
Eigen::Vector3f QuadraticFormula(double A, double B, double C) {
    double discriminate = B*B - 4*A*C;

    if (discriminate < 0) {
        return Eigen::Vector3f(0,0,0);
    }

    double sqrtDisc = sqrt(discriminate);

    float plusOp = (-B + sqrtDisc)/(2*A);

    if (discriminate == 0) {
        return Eigen::Vector3f(1, plusOp, 0);
    }

    float minOp = (-B - sqrtDisc)/(2*A);

    return Eigen::Vector3f(2, plusOp, minOp);
}
