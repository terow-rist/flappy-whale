#include "utils.h"
#include <string.h>
static float deg2rad(float d){ return d * 3.14159265358979323846f / 180.0f; }
mat4 mat4_identity(){ mat4 o; memset(o.m,0,sizeof(o.m)); o.m[0]=o.m[5]=o.m[10]=o.m[15]=1.0f; return o;}
mat4 mat4_perspective(float fovy_deg, float aspect, float nearp, float farp){
    float f = 1.0f / tanf(deg2rad(fovy_deg)/2.0f);
    mat4 o; memset(o.m,0,sizeof(o.m));
    o.m[0] = f / aspect;
    o.m[5] = f;
    o.m[10] = (farp + nearp) / (nearp - farp);
    o.m[11] = -1.0f;
    o.m[14] = (2.0f * farp * nearp) / (nearp - farp);
    return o;
}
mat4 mat4_translate(float x, float y, float z){
    mat4 o = mat4_identity(); o.m[12]=x; o.m[13]=y; o.m[14]=z; return o;
}
mat4 mat4_scale(float x, float y, float z){
    mat4 o; memset(o.m,0,sizeof(o.m)); o.m[0]=x; o.m[5]=y; o.m[10]=z; o.m[15]=1.0f; return o;
}
mat4 mat4_mul(mat4 a, mat4 b){
    mat4 r; for(int i=0;i<4;i++) for(int j=0;j<4;j++){ r.m[i*4+j]=0; for(int k=0;k<4;k++) r.m[i*4+j]+= a.m[i*4+k]*b.m[k*4+j]; } return r;
}
mat4 mat4_lookat(float ex, float ey, float ez, float cx, float cy, float cz, float ux, float uy, float uz){
    float fx = cx-ex, fy = cy-ey, fz = cz-ez;
    float rlf = 1.0f / sqrtf(fx*fx+fy*fy+fz*fz);
    fx*=rlf; fy*=rlf; fz*=rlf;
    float sx = fy*uz - fz*uy;
    float sy = fz*ux - fx*uz;
    float sz = fx*uy - fy*ux;
    float rls = 1.0f / sqrtf(sx*sx+sy*sy+sz*sz);
    sx*=rls; sy*=rls; sz*=rls;
    float ux2 = sy*fz - sz*fy;
    float uy2 = sz*fx - sx*fz;
    float uz2 = sx*fy - sy*fx;
    mat4 o; memset(o.m,0,sizeof(o.m));
    o.m[0]=sx; o.m[1]=ux2; o.m[2]=-fx; o.m[3]=0;
    o.m[4]=sy; o.m[5]=uy2; o.m[6]=-fy; o.m[7]=0;
    o.m[8]=sz; o.m[9]=uz2; o.m[10]=-fz; o.m[11]=0;
    o.m[12]=-(sx*ex + sy*ey + sz*ez);
    o.m[13]=-(ux2*ex + uy2*ey + uz2*ez);
    o.m[14]= fx*ex + fy*ey + fz*ez;
    o.m[15]=1;
    return o;
}
