#ifndef UTILS_H
#define UTILS_H
#include <math.h>
typedef struct { float m[16]; } mat4;
mat4 mat4_identity();
mat4 mat4_perspective(float fovy_deg, float aspect, float nearp, float farp);
mat4 mat4_translate(float x, float y, float z);
mat4 mat4_scale(float x, float y, float z);
mat4 mat4_mul(mat4 a, mat4 b);
mat4 mat4_lookat(float ex, float ey, float ez, float cx, float cy, float cz, float ux, float uy, float uz);
#endif
