#version 120

struct Material {
  vec3 aColor;
  vec3 dColor;
  vec3 sColor;
  float shine;
};

uniform Material uMat;
uniform float isDot;

varying vec3 fragNor;
varying vec4 light;
varying float dist;
varying vec3 fragColor;

void main()
{
	if (isDot > 0) {
		gl_FragColor = vec4(fragColor, 1.0);
	} else {
		vec3 lightPos = vec3(0,1,0);
		float dotNormLigh;
		vec3 lightColor = vec3(1.5,1.5,1.5);
		vec3 normal = normalize(fragNor); //change normal back to vec3 if you want it to go back
		// Map normal in the range [-1, 1] to color in range [0, 1];
		vec3 color;							// = 0.5*normal + 0.5;

		dotNormLigh = clamp(dot(normal, lightPos), 0.0, 1.0);

		color = dist * (lightColor * uMat.dColor * dotNormLigh) + uMat.aColor*lightColor;

		gl_FragColor = vec4(color, 1.0);
	}
}
