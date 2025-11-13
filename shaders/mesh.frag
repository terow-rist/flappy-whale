#version 330 core
in vec3 vNormal;
in vec3 vPos;
out vec4 FragColor;
uniform vec3 uLightPos;
uniform vec3 uColor;
void main(){
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightPos - vPos);
    float diff = max(dot(N,L), 0.0);
    vec3 col = uColor * (0.2 + 0.8*diff);
    FragColor = vec4(col,1.0);
}
