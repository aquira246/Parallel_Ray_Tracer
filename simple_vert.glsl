#version 120

struct Material {
  vec3 aColor;
  vec3 dColor;
  vec3 sColor;
  float shine;
};

attribute vec3 vertPos;
attribute vec3 vertNor;

uniform mat4 P;		//projection matrix
uniform mat4 MV;	//model-view matrix
uniform Material uMat;
uniform float isDot;

varying vec3 fragNor;
varying vec4 light;
varying float dist;
varying vec3 fragColor;


void main()
{
	if (isDot > 0) {
		gl_Position = P * MV * gl_Vertex;
		fragColor = gl_Color.rgb;
	} else {
		vec4 lightPos = vec4(0,1,0,1);
		/*set up lighting before camera and after translations*/
		vec4 newPos = vec4(vertPos, 1.0);
		light = lightPos - newPos; //the vector of light position - v position
		light = normalize(light);

		/*calculate reflection*/
		//Refl = vec3(-light + 2.0*(clamp(dot(tNormal, light), 0.0, 1.0))*tNormal);
		  
		/*calculate the distance*/
		dist = distance(newPos, lightPos);
		dist = (1.0/(dist));

		//no reflection yet

		gl_Position = P * MV * newPos;
		fragNor = (MV * vec4(vertNor, 0.0)).xyz;
	}
}
