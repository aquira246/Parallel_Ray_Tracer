#include <Shape.hpp>
#include "Helper.h"

using namespace std;

Shape::Shape() {
    SetMaterialByNum(rand() % NUM_MATS);
}

Shape::~Shape(){

}

void Shape::SetMaterialToMat(Material newMat) {
    mat = newMat;
}

void Shape::SetMaterialByNum(int colorNum) {
    Eigen::Vector3f a, s, d;
    float shine;

    switch (colorNum) {
        case 0:
            a = Eigen::Vector3f(.2, .2, .2);
            d = Eigen::Vector3f(.8, .2, .2);
            s = Eigen::Vector3f(.4, .4, .4);
            shine = 200;
            break;
        
        case 1:
            a = Eigen::Vector3f(.2, .2, .2);
            d = Eigen::Vector3f(.2, .2, .8);
            s = Eigen::Vector3f(.4, .4, .4);
            shine = 200;
            break;

        case 2:
            a = Eigen::Vector3f(.2, .2, .2);
            d = Eigen::Vector3f(.2, .8, .2);
            s = Eigen::Vector3f(.4, .4, .4);
            shine = 200;
            break;

        case 3:
            a = Eigen::Vector3f(.2, .2, .2);
            d = Eigen::Vector3f(.8, .2, .8);
            s = Eigen::Vector3f(.4, .4, .4);
            shine = 200;
            break;

        case 4:
            a = Eigen::Vector3f(.2, .2, .2);
            d = Eigen::Vector3f(.2, .8, .8);
            s = Eigen::Vector3f(.4, .4, .4);
            shine = 200;
            break;

        case 5:
            a = Eigen::Vector3f(.2, .2, .2);
            d = Eigen::Vector3f(.8, .8, .2);
            s = Eigen::Vector3f(.4, .4, .4);
            shine = 200;
            break;

        default:
            a = Eigen::Vector3f(.2, .2, .2);
            d = Eigen::Vector3f(.6, .6, .6);
            s = Eigen::Vector3f(.6, .6, .6);
            shine = 260;
            break;
    }

    mat.ambient = a;
    mat.diffuse = d;
    mat.specular = s;
    mat.shine = shine;
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
