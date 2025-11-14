// src/flappy_whale.cpp
// Flappy Whale 3D — full source with fullscreen window, camera follow, HUD (score bar + FPS),
// stalactites matching hitboxes, top inverted/up, bottom inverted/down.
// Replace your existing file with this.

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <string>
#include <limits>
#include <filesystem>
#include <cstring>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

// optional for Windows headers warning cleanup (won't hurt if not windows)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

int WIN_W = 1280;
int WIN_H = 720;

const float GRAVITY = -1500.0f;
const float JUMP_V = 480.0f;
const float WHALE_RADIUS = 28.0f;
const float PIPE_HALF_WIDTH = 40.0f;
const float GAP_HEIGHT = 15.0f;
const float PIPE_DEPTH = 160.0f;
const float PIPE_SPEED = 420.0f;
const int INITIAL_PIPES = 6;
const float PIPE_SPACING_Z = 450.0f;
const float SPAWN_Z = 1800.0f;
const float DESPAWN_Z = -200.0f;

// Keep hitbox shrink factor (visuals will match these reduced hitboxes)
const float HITBOX_SCALE = 0.55f;
// global model scale multiplier
const float GLOBAL_STAL_SCALE = 1.0f;

constexpr float PI = 3.14159265358979323846f;

static std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
int rand_int(int a,int b){ std::uniform_int_distribution<int> d(a,b); return d(rng); }

struct Mat4 { float m[16]; };
Mat4 mat4_identity(){ Mat4 M; for(int i=0;i<16;++i) M.m[i]=0.0f; M.m[0]=M.m[5]=M.m[10]=M.m[15]=1.0f; return M; }
Mat4 mat4_mul(const Mat4 &a, const Mat4 &b){
    Mat4 r; for(int i=0;i<16;++i) r.m[i]=0.0f;
    for(int row=0;row<4;++row) for(int col=0;col<4;++col){
        float s=0.f;
        for(int k=0;k<4;++k) s += a.m[k*4 + row] * b.m[col*4 + k];
        r.m[col*4 + row] = s;
    }
    return r;
}
Mat4 mat4_translate(float x,float y,float z){
    Mat4 M = mat4_identity(); M.m[12]=x; M.m[13]=y; M.m[14]=z; return M;
}
Mat4 mat4_scale(float sx,float sy,float sz){
    Mat4 M; for(int i=0;i<16;++i) M.m[i]=0.0f;
    M.m[0]=sx; M.m[5]=sy; M.m[10]=sz; M.m[15]=1.0f; return M;
}
Mat4 mat4_perspective(float fovy,float aspect,float zn,float zf){
    float f = 1.0f / tanf(fovy*0.5f);
    Mat4 M; for(int i=0;i<16;++i) M.m[i]=0.f;
    M.m[0] = f/aspect; M.m[5]=f; M.m[10]=(zf+zn)/(zn-zf); M.m[11]=-1.0f; M.m[14]=(2.f*zf*zn)/(zn-zf);
    return M;
}
Mat4 mat4_lookat(float ex,float ey,float ez, float cx,float cy,float cz, float ux,float uy,float uz){
    float fx = cx-ex, fy = cy-ey, fz = cz-ez;
    float fl = sqrtf(fx*fx+fy*fy+fz*fz); if(fl==0) fl=1; fx/=fl; fy/=fl; fz/=fl;
    float rx = fy*uz - fz*uy;
    float ry = fz*ux - fx*uz;
    float rz = fx*uy - fy*ux;
    float rl = sqrtf(rx*rx+ry*ry+rz*rz); if(rl==0) rl=1; rx/=rl; ry/=rl; rz/=rl;
    float ux2 = ry*fz - rz*fy;
    float uy2 = rz*fx - rx*fz;
    float uz2 = rx*fy - ry*fx;
    Mat4 M = mat4_identity();
    M.m[0]=rx; M.m[4]=ry; M.m[8]=rz;
    M.m[1]=ux2; M.m[5]=uy2; M.m[9]=uz2;
    M.m[2]=-fx; M.m[6]=-fy; M.m[10]=-fz;
    M.m[12] = -(rx*ex + ry*ey + rz*ez);
    M.m[13] = -(ux2*ex + uy2*ey + uz2*ez);
    M.m[14] =  (fx*ex + fy*ey + fz*ez);
    return M;
}
Mat4 mat4_rotate_x(float a){
    float c = cosf(a), s = sinf(a);
    Mat4 M; for(int i=0;i<16;++i) M.m[i]=0.0f;
    M.m[0]=1.0f; M.m[5]=c; M.m[6]=s; M.m[9]=-s; M.m[10]=c; M.m[15]=1.0f;
    return M;
}

// shaders
const char *vertex_deform_src = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(std140) uniform Global { mat4 uPV; vec4 uLightDir; vec4 uCamPos; };
uniform mat4 uModel;
out vec3 vWorldPos;
out vec3 vNormal;
void main(){
    vec4 worldPos4 = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos4.xyz;
    mat3 normalMatrix = mat3(uModel);
    vNormal = normalize(normalMatrix * aNormal);
    gl_Position = uPV * worldPos4;
}
)";
const char *fragment_phong_src = R"(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
out vec4 FragColor;
layout(std140) uniform Global { mat4 uPV; vec4 uLightDir; vec4 uCamPos; };
uniform vec3 uColor;
uniform float uSpecularPower;
uniform float uSpecularIntensity;
void main(){
    vec3 N = normalize(vNormal);
    vec3 L = normalize(-uLightDir.xyz);
    vec3 V = normalize(uCamPos.xyz - vWorldPos);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = 0.0;
    if(diff > 0.0) spec = pow(max(dot(N, H), 0.0), uSpecularPower) * uSpecularIntensity;
    vec3 base = uColor;
    vec3 ambient = base * 0.18;
    vec3 color = ambient + base * diff + vec3(spec);
    color = pow(color, vec3(1.0/2.2));
    FragColor = vec4(color, 1.0);
}
)";

const char *ov_vs = R"(
#version 330 core
layout(location=0) in vec2 aPos;
void main(){ gl_Position = vec4(aPos,0.0,1.0); }
)";
const char *ov_fs = R"(
#version 330 core
uniform vec3 uColor;
out vec4 F;
void main(){ F = vec4(uColor,1.0); }
)";

struct Model { GLuint vao=0, vbo=0, ebo=0; GLsizei idxCount=0; float minx, miny, minz, maxx, maxy, maxz, centerx, centery, centerz; bool valid=false; };
void destroy_model(Model &m){ if(m.vbo) glDeleteBuffers(1,&m.vbo); if(m.ebo) glDeleteBuffers(1,&m.ebo); if(m.vao) glDeleteVertexArrays(1,&m.vao); m=Model(); }
void model_half_extents(const Model &m, float &hx, float &hy, float &hz){ hx = (m.maxx - m.minx) * 0.5f; hy = (m.maxy - m.miny) * 0.5f; hz = (m.maxz - m.minz) * 0.5f; }

struct Mesh { GLuint vao=0, vbo=0, ebo=0; GLsizei idxCount=0; };
Mesh make_cube_mesh(){
    Mesh m;
    struct V{ float x,y,z; float nx,ny,nz; };
    V verts[] = {
        {-1,-1, 1, 0,0,1},{1,-1, 1, 0,0,1},{1, 1, 1, 0,0,1},{-1, 1, 1, 0,0,1},
        {-1,-1,-1, 0,0,-1},{-1, 1,-1, 0,0,-1},{1, 1,-1, 0,0,-1},{1,-1,-1, 0,0,-1},
        {-1,-1,-1, -1,0,0},{-1,-1, 1, -1,0,0},{-1, 1, 1, -1,0,0},{-1, 1,-1, -1,0,0},
        {1,-1,-1, 1,0,0},{1, 1,-1, 1,0,0},{1, 1, 1, 1,0,0},{1,-1, 1, 1,0,0},
        {-1, 1,-1, 0,1,0},{-1, 1, 1, 0,1,0},{1, 1, 1, 0,1,0},{1, 1,-1, 0,1,0},
        {-1,-1,-1, 0,-1,0},{1,-1,-1, 0,-1,0},{1,-1, 1, 0,-1,0},{-1,-1, 1, 0,-1,0}
    };
    unsigned int idxs[] = {
        0,1,2, 0,2,3,
        4,5,6, 4,6,7,
        8,9,10, 8,10,11,
        12,13,14, 12,14,15,
        16,17,18, 16,18,19,
        20,21,22, 20,22,23
    };
    glGenVertexArrays(1,&m.vao);
    glBindVertexArray(m.vao);
    glGenBuffers(1,&m.vbo);
    glBindBuffer(GL_ARRAY_BUFFER,m.vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_STATIC_DRAW);
    glGenBuffers(1,&m.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(idxs),idxs,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(V),(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(V),(void*)(3*sizeof(float)));
    glBindVertexArray(0);
    m.idxCount = sizeof(idxs)/sizeof(idxs[0]);
    return m;
}

const char *water_vs = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uPV;
uniform mat4 uModel;
uniform float uTime;
out vec3 vWorldPos;
out vec3 vNormal;
void main(){
    float wave = sin(aPos.x * 0.02 + uTime * 1.2) * 6.0
               + sin(aPos.z * 0.015 + uTime * 1.6) * 3.5;
    vec3 pos = aPos + vec3(0.0, wave, 0.0);
    float dx = cos(aPos.x * 0.02 + uTime * 1.2) * 0.02 * 6.0;
    float dz = cos(aPos.z * 0.015 + uTime * 1.6) * 0.015 * 3.5;
    vec3 N = normalize(vec3(-dx, 1.0, -dz));
    vNormal = N;
    vec4 world = uModel * vec4(pos,1.0);
    vWorldPos = world.xyz;
    gl_Position = uPV * world;
}
)";
const char *water_fs = R"(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
out vec4 FragColor;
uniform vec3 uCameraPos;
uniform vec3 uLightDir;
uniform float uTime;
void main(){
    vec3 N = normalize(vNormal);
    vec3 L = normalize(-uLightDir);
    vec3 V = normalize(uCameraPos - vWorldPos);
    float diff = max(dot(N,L), 0.0);
    float fres = pow(1.0 - max(dot(N,V),0.0), 3.0) * 0.9;
    vec3 base = vec3(0.02, 0.30, 0.50);
    vec3 shallow = vec3(0.15,0.55,0.85);
    float depthFactor = clamp((vWorldPos.y + 80.0)/200.0, 0.0, 1.0);
    vec3 col = mix(base, shallow, depthFactor);
    vec3 color = col * (0.15 + diff*0.9) + fres * vec3(1.0,1.0,1.0)*0.4;
    FragColor = vec4(pow(color, vec3(1.0/2.2)), 0.9);
}
)";

Mesh make_water_grid(int nx=200, int nz=200, float sx=1600.0f, float sz=1200.0f){
    std::vector<float> v;
    std::vector<unsigned int> idx;
    v.reserve((nx+1)*(nz+1)*6);
    for(int iz=0; iz<=nz; ++iz){
        for(int ix=0; ix<=nx; ++ix){
            float x = ((float)ix / (float)nx - 0.5f) * sx;
            float z = ((float)iz / (float)nz - 0.5f) * sz;
            v.push_back(x); v.push_back(0.0f); v.push_back(z);
            v.push_back(0.0f); v.push_back(1.0f); v.push_back(0.0f);
        }
    }
    for(int iz=0; iz<nz; ++iz){
        for(int ix=0; ix<nx; ++ix){
            int a = iz*(nx+1) + ix;
            int b = a + 1;
            int c = a + (nx+1);
            int d = c + 1;
            idx.push_back(a); idx.push_back(c); idx.push_back(b);
            idx.push_back(b); idx.push_back(c); idx.push_back(d);
        }
    }
    Mesh m;
    glGenVertexArrays(1,&m.vao);
    glBindVertexArray(m.vao);
    glGenBuffers(1,&m.vbo);
    glBindBuffer(GL_ARRAY_BUFFER,m.vbo);
    glBufferData(GL_ARRAY_BUFFER, v.size()*sizeof(float), v.data(), GL_STATIC_DRAW);
    glGenBuffers(1,&m.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size()*sizeof(unsigned int), idx.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));
    glBindVertexArray(0);
    m.idxCount = (GLsizei)idx.size();
    return m;
}

// tinygltf loader (same as before)
static inline size_t component_size(int componentType){
    switch(componentType){
        case TINYGLTF_COMPONENT_TYPE_BYTE: return 1;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: return 1;
        case TINYGLTF_COMPONENT_TYPE_SHORT: return 2;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return 2;
        case TINYGLTF_COMPONENT_TYPE_INT: return 4;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: return 4;
        case TINYGLTF_COMPONENT_TYPE_FLOAT: return 4;
        default: return 0;
    }
}
static inline int num_components(int type){
    switch(type){
        case TINYGLTF_TYPE_SCALAR: return 1;
        case TINYGLTF_TYPE_VEC2: return 2;
        case TINYGLTF_TYPE_VEC3: return 3;
        case TINYGLTF_TYPE_VEC4: return 4;
        case TINYGLTF_TYPE_MAT2: return 4;
        case TINYGLTF_TYPE_MAT3: return 9;
        case TINYGLTF_TYPE_MAT4: return 16;
        default: return 0;
    }
}

bool load_gltf_model(const std::string &path, Model &out){
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ok = false;
    std::string ext;
    size_t p = path.find_last_of('.');
    if(p != std::string::npos) ext = path.substr(p+1);
    if(ext == "glb" || ext == "GLB") {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    } else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }
    if(!warn.empty()) std::cerr<<"tinygltf warn: "<<warn<<"\n";
    if(!err.empty()) std::cerr<<"tinygltf err: "<<err<<"\n";
    if(!ok) return false;

    auto node_local_matrix = [&](const tinygltf::Node &node){
        std::array<float,16> M;
        for(int i=0;i<16;i++) M[i]=0.0f;
        if(!node.matrix.empty()){
            for(int i=0;i<16 && i<(int)node.matrix.size(); ++i) M[i] = (float)node.matrix[i];
        } else {
            float tx=0,ty=0,tz=0;
            if(!node.translation.empty()){ tx = (float)node.translation[0]; ty = (float)node.translation[1]; tz = (float)node.translation[2]; }
            float qx=0,qy=0,qz=0,qw=1;
            if(!node.rotation.empty()){ qx=(float)node.rotation[0]; qy=(float)node.rotation[1]; qz=(float)node.rotation[2]; qw=(float)node.rotation[3]; }
            float sx=1,sy=1,sz=1;
            if(!node.scale.empty()){ sx=(float)node.scale[0]; sy=(float)node.scale[1]; sz=(float)node.scale[2]; }
            float x2=qx+qx, y2=qy+qy, z2=qz+qz;
            float xx=qx*x2, yy=qy*y2, zz=qz*z2;
            float xy=qx*y2, xz=qx*z2, yz=qy*z2;
            float wx=qw*x2, wy=qw*y2, wz=qw*z2;
            float m00 = 1.0f - (yy + zz);
            float m01 = xy + wz;
            float m02 = xz - wy;
            float m10 = xy - wz;
            float m11 = 1.0f - (xx + zz);
            float m12 = yz + wx;
            float m20 = xz + wy;
            float m21 = yz - wx;
            float m22 = 1.0f - (xx + yy);
            m00 *= sx; m01 *= sx; m02 *= sx;
            m10 *= sy; m11 *= sy; m12 *= sy;
            m20 *= sz; m21 *= sz; m22 *= sz;
            M[0]=m00; M[1]=m10; M[2]=m20; M[3]=0.0f;
            M[4]=m01; M[5]=m11; M[6]=m21; M[7]=0.0f;
            M[8]=m02; M[9]=m12; M[10]=m22; M[11]=0.0f;
            M[12]=tx; M[13]=ty; M[14]=tz; M[15]=1.0f;
        }
        return M;
    };

    std::vector<std::array<float,16>> nodeWorld(model.nodes.size());
    std::function<void(int,const std::array<float,16>&)> computeWorld;
    std::array<float,16> I;
    for(int i=0;i<16;i++){ I[i] = (i%5==0) ? 1.0f : 0.0f; }
    computeWorld = [&](int nodeIdx, const std::array<float,16> &parent){
        const tinygltf::Node &node = model.nodes[nodeIdx];
        std::array<float,16> local = node_local_matrix(node);
        std::array<float,16> W;
        for(int col=0;col<4;++col){
            for(int row=0;row<4;++row){
                float s=0.0f;
                for(int k=0;k<4;++k) s += parent[k*4 + row] * local[col*4 + k];
                W[col*4 + row] = s;
            }
        }
        nodeWorld[nodeIdx] = W;
        for(int c : node.children){
            computeWorld(c, W);
        }
    };
    std::vector<int> parent(model.nodes.size(), -1);
    for(size_t i=0;i<model.nodes.size();++i){
        for(int c : model.nodes[i].children){
            if(c >= 0 && c < (int)model.nodes.size()) parent[c] = (int)i;
        }
    }
    for(size_t i=0;i<model.nodes.size();++i){
        if(parent[i] == -1){
            computeWorld((int)i, I);
        }
    }

    std::vector<float> allVerts;
    std::vector<unsigned int> allIdx;
    float minx=FLT_MAX, miny=FLT_MAX, minz=FLT_MAX;
    float maxx=-FLT_MAX, maxy=-FLT_MAX, maxz=-FLT_MAX;

    auto transform_pos = [](const std::array<float,16> &M, float x,float y,float z){
        float rx = M[0]*x + M[4]*y + M[8]*z + M[12];
        float ry = M[1]*x + M[5]*y + M[9]*z + M[13];
        float rz = M[2]*x + M[6]*y + M[10]*z + M[14];
        return std::array<float,3>{rx,ry,rz};
    };
    auto transform_normal = [](const std::array<float,16> &M, float nx,float ny,float nz){
        float m00=M[0], m01=M[4], m02=M[8];
        float m10=M[1], m11=M[5], m12=M[9];
        float m20=M[2], m21=M[6], m22=M[10];
        float rx = m00*nx + m01*ny + m02*nz;
        float ry = m10*nx + m11*ny + m12*nz;
        float rz = m20*nx + m21*ny + m22*nz;
        float len = sqrtf(rx*rx + ry*ry + rz*rz);
        if(len < 1e-6f) return std::array<float,3>{0.0f,1.0f,0.0f};
        return std::array<float,3>{rx/len, ry/len, rz/len};
    };

    for(size_t ni=0; ni<model.nodes.size(); ++ni){
        const tinygltf::Node &node = model.nodes[ni];
        if(node.mesh < 0) continue;
        std::array<float,16> W = nodeWorld[ni];
        const tinygltf::Mesh &mesh = model.meshes[node.mesh];
        for(const auto &prim : mesh.primitives){
            auto itPos = prim.attributes.find("POSITION");
            if(itPos == prim.attributes.end()) continue;
            int posAccessorIdx = itPos->second;
            const tinygltf::Accessor &posAcc = model.accessors[posAccessorIdx];
            const tinygltf::BufferView &posView = model.bufferViews[posAcc.bufferView];
            const tinygltf::Buffer &posBuffer = model.buffers[posView.buffer];

            bool hasNormal = (prim.attributes.find("NORMAL") != prim.attributes.end());
            int norAccessorIdx = hasNormal ? prim.attributes.at("NORMAL") : -1;
            const tinygltf::Accessor *norAcc = hasNormal ? &model.accessors[norAccessorIdx] : nullptr;
            const tinygltf::BufferView *norView = hasNormal ? &model.bufferViews[norAcc->bufferView] : nullptr;

            std::vector<unsigned int> indices;
            if(prim.indices >= 0){
                const tinygltf::Accessor &ia = model.accessors[prim.indices];
                const tinygltf::BufferView &ibv = model.bufferViews[ia.bufferView];
                const tinygltf::Buffer &ib = model.buffers[ibv.buffer];
                const unsigned char* idata = ib.data.data() + ibv.byteOffset + ia.byteOffset;
                indices.resize(ia.count);
                if(ia.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT){
                    const unsigned short* s = reinterpret_cast<const unsigned short*>(idata);
                    for(size_t i=0;i<ia.count;i++) indices[i] = (unsigned int)s[i];
                } else if(ia.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT){
                    const unsigned int* s = reinterpret_cast<const unsigned int*>(idata);
                    for(size_t i=0;i<ia.count;i++) indices[i] = s[i];
                } else if(ia.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE){
                    const unsigned char* s = idata;
                    for(size_t i=0;i<ia.count;i++) indices[i] = (unsigned int)s[i];
                } else {
                    continue;
                }
            } else {
                indices.clear();
            }

            size_t vc = posAcc.count;
            size_t baseVertexIndex = allVerts.size() / 6;
            const unsigned char* basePos = posBuffer.data.data() + posView.byteOffset + posAcc.byteOffset;
            size_t posStride = posView.byteStride ? posView.byteStride : (component_size(posAcc.componentType) * num_components(posAcc.type));
            const unsigned char* baseNor = nullptr;
            size_t norStride = 0;
            if(hasNormal){
                const tinygltf::Buffer &nb = model.buffers[norView->buffer];
                baseNor = nb.data.data() + norView->byteOffset + norAcc->byteOffset;
                norStride = norView->byteStride ? norView->byteStride : (component_size(norAcc->componentType) * num_components(norAcc->type));
            }

            for(size_t vi=0; vi<vc; ++vi){
                const unsigned char* pptr = basePos + vi * posStride;
                float px,py,pz;
                std::memcpy(&px, pptr + 0, sizeof(float));
                std::memcpy(&py, pptr + 4, sizeof(float));
                std::memcpy(&pz, pptr + 8, sizeof(float));
                auto tp = transform_pos(W, px,py,pz);
                float nx=0, ny=1, nz=0;
                if(hasNormal){
                    const unsigned char* nptr = baseNor + vi * norStride;
                    float inx,iny,inz;
                    std::memcpy(&inx, nptr + 0, sizeof(float));
                    std::memcpy(&iny, nptr + 4, sizeof(float));
                    std::memcpy(&inz, nptr + 8, sizeof(float));
                    auto tn = transform_normal(W, inx,iny,inz);
                    nx = tn[0]; ny = tn[1]; nz = tn[2];
                } else {
                    nx = 0; ny = 0; nz = 0;
                }
                allVerts.push_back(tp[0]); allVerts.push_back(tp[1]); allVerts.push_back(tp[2]);
                allVerts.push_back(nx); allVerts.push_back(ny); allVerts.push_back(nz);
                if(tp[0] < minx) minx = tp[0]; if(tp[1] < miny) miny = tp[1]; if(tp[2] < minz) minz = tp[2];
                if(tp[0] > maxx) maxx = tp[0]; if(tp[1] > maxy) maxy = tp[1]; if(tp[2] > maxz) maxz = tp[2];
            }

            if(prim.indices >= 0){
                for(size_t i=0;i<indices.size();++i) allIdx.push_back(indices[i] + (unsigned int)baseVertexIndex);
            } else {
                for(unsigned int i=0;i<vc;i++) allIdx.push_back((unsigned int)(i + baseVertexIndex));
            }
        }
    }

    if(allVerts.empty() || allIdx.empty()) return false;

    bool needNormals = false;
    for(size_t i=0;i<allVerts.size()/6;i++){
        float nx = allVerts[i*6+3], ny = allVerts[i*6+4], nz = allVerts[i*6+5];
        if(fabs(nx) < 1e-6f && fabs(ny) < 1e-6f && fabs(nz) < 1e-6f) { needNormals = true; break; }
    }
    if(needNormals){
        std::vector<float> accum((allVerts.size()/6)*3, 0.0f);
        for(size_t i=0;i+2<allIdx.size(); i+=3){
            unsigned int i0 = allIdx[i+0], i1 = allIdx[i+1], i2 = allIdx[i+2];
            float p0x = allVerts[i0*6+0], p0y = allVerts[i0*6+1], p0z = allVerts[i0*6+2];
            float p1x = allVerts[i1*6+0], p1y = allVerts[i1*6+1], p1z = allVerts[i1*6+2];
            float p2x = allVerts[i2*6+0], p2y = allVerts[i2*6+1], p2z = allVerts[i2*6+2];
            float ux = p1x - p0x, uy = p1y - p0y, uz = p1z - p0z;
            float vx = p2x - p0x, vy = p2y - p0y, vz = p2z - p0z;
            float cx = uy * vz - uz * vy;
            float cy = uz * vx - ux * vz;
            float cz = ux * vy - uy * vx;
            accum[i0*3+0] += cx; accum[i0*3+1] += cy; accum[i0*3+2] += cz;
            accum[i1*3+0] += cx; accum[i1*3+1] += cy; accum[i1*3+2] += cz;
            accum[i2*3+0] += cx; accum[i2*3+1] += cy; accum[i2*3+2] += cz;
        }
        for(size_t vi=0; vi<allVerts.size()/6; ++vi){
            float nx = accum[vi*3+0], ny = accum[vi*3+1], nz = accum[vi*3+2];
            float len = sqrtf(nx*nx + ny*ny + nz*nz);
            if(len < 1e-6f){ nx = 0; ny = 1; nz = 0; }
            else { nx/=len; ny/=len; nz/=len; }
            allVerts[vi*6+3] = nx; allVerts[vi*6+4] = ny; allVerts[vi*6+5] = nz;
        }
    }

    float cx_b = (minx + maxx) * 0.5f;
    float cy_b = (miny + maxy) * 0.5f;
    float cz_b = (minz + maxz) * 0.5f;
    for(size_t vi=0; vi<allVerts.size()/6; ++vi){
        allVerts[vi*6+0] -= cx_b;
        allVerts[vi*6+1] -= cy_b;
        allVerts[vi*6+2] -= cz_b;
    }
    float nminx = minx - cx_b, nmaxx = maxx - cx_b;
    float nminy = miny - cy_b, nmaxy = maxy - cy_b;
    float nminz = minz - cz_b, nmaxz = maxz - cz_b;

    glGenVertexArrays(1,&out.vao);
    glBindVertexArray(out.vao);
    glGenBuffers(1,&out.vbo);
    glBindBuffer(GL_ARRAY_BUFFER,out.vbo);
    glBufferData(GL_ARRAY_BUFFER, allVerts.size()*sizeof(float), allVerts.data(), GL_STATIC_DRAW);
    glGenBuffers(1,&out.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,out.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, allIdx.size()*sizeof(unsigned int), allIdx.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));
    glBindVertexArray(0);

    out.idxCount = (GLsizei)allIdx.size();
    out.minx = nminx; out.miny = nminy; out.minz = nminz;
    out.maxx = nmaxx; out.maxy = nmaxy; out.maxz = nmaxz;
    out.centerx = cx_b; out.centery = cy_b; out.centerz = cz_b;
    out.valid = true;
    return true;
}

// game state
struct Pipe { float x,y,z; bool passed; int id; Pipe(float X,float Y,float Z,int Id):x(X),y(Y),z(Z),passed(false),id(Id){} };
std::vector<Pipe> pipes;
static int g_nextPipeId = 0;

struct Whale { float x,y,z; float vy; bool alive; Whale(){ x = 0.0f; y = WIN_H*0.5f; z = 0.0f; vy = 0.0f; alive = true; } void reset(){ y = WIN_H*0.5f; vy = 0.0f; alive = true; } } whale;

int score=0, highScore=0;
bool wantJump=false;
bool showHitboxes=false;

static GLuint g_globalUBO = 0;

void key_cb(GLFWwindow* w,int key,int sc,int action,int mods){
    if(key==GLFW_KEY_SPACE && action==GLFW_PRESS) wantJump=true;
    if(key==GLFW_KEY_E && action==GLFW_PRESS) showHitboxes = !showHitboxes;
    if(key==GLFW_KEY_ESCAPE && action==GLFW_PRESS) glfwSetWindowShouldClose(w,GLFW_TRUE);
}

struct AABB { float cx,cy,cz; float hx,hy,hz; };

bool sphere_aabb_collide(float sx,float sy,float sz,float r, const AABB &b){
    float dx = std::fmax(b.cx - b.hx, std::fmin(sx, b.cx + b.hx)) - sx;
    float dy = std::fmax(b.cy - b.hy, std::fmin(sy, b.cy + b.hy)) - sy;
    float dz = std::fmax(b.cz - b.hz, std::fmin(sz, b.cz + b.hz)) - sz;
    float dist2 = dx*dx + dy*dy + dz*dz;
    return dist2 <= r*r;
}

struct ScreenRect { float x0,y0,x1,y1; };
bool world_to_screen(const Mat4 &PV, float wx,float wy,float wz, float &sx,float &sy,float &sz){
    float x = PV.m[0]*wx + PV.m[4]*wy + PV.m[8]*wz + PV.m[12]*1.0f;
    float y = PV.m[1]*wx + PV.m[5]*wy + PV.m[9]*wz + PV.m[13]*1.0f;
    float z = PV.m[2]*wx + PV.m[6]*wy + PV.m[10]*wz + PV.m[14]*1.0f;
    float w = PV.m[3]*wx + PV.m[7]*wy + PV.m[11]*wz + PV.m[15]*1.0f;
    if(fabs(w) < 1e-6f) return false;
    float ndc_x = x/w, ndc_y = y/w, ndc_z = z/w;
    sx = (ndc_x * 0.5f + 0.5f) * WIN_W;
    sy = (1.0f - (ndc_y * 0.5f + 0.5f)) * WIN_H;
    sz = ndc_z;
    return true;
}
ScreenRect aabb_to_screen_rect(const Mat4 &PV, const AABB &b){
    float minx=1e9f,miny=1e9f,maxx=-1e9f,maxy=-1e9f;
    for(int ix=-1;ix<=1;ix+=2) for(int iy=-1;iy<=1;iy+=2) for(int iz=-1;iz<=1;iz+=2){
        float cx = b.cx + ix * b.hx;
        float cy = b.cy + iy * b.hy;
        float cz = b.cz + iz * b.hz;
        float sx,sy,sz;
        if(world_to_screen(PV,cx,cy,cz,sx,sy,sz)){
            minx = std::min(minx, sx);
            miny = std::min(miny, sy);
            maxx = std::max(maxx, sx);
            maxy = std::max(maxy, sy);
        }
    }
    if(minx>maxx) return {0,0,0,0};
    return {minx,miny,maxx,maxy};
}

GLuint ovProg=0; GLint ov_uColor=-1;
void ensure_overlay(){
    if(ovProg) return;
    GLuint vs = glCreateShader(GL_VERTEX_SHADER); const char* src = ov_vs; glShaderSource(vs,1,&src,nullptr); glCompileShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER); const char* src2 = ov_fs; glShaderSource(fs,1,&src2,nullptr); glCompileShader(fs);
    ovProg = glCreateProgram(); glAttachShader(ovProg,vs); glAttachShader(ovProg,fs); glLinkProgram(ovProg);
    glDeleteShader(vs); glDeleteShader(fs);
    ov_uColor = glGetUniformLocation(ovProg, "uColor");
}
inline float px_to_ndc_x(float px){ return (px / (float)WIN_W) * 2.0f - 1.0f; }
inline float px_to_ndc_y(float py){ return 1.0f - (py / (float)WIN_H) * 2.0f; }
void draw_screen_rect_lines(float x0,float y0,float x1,float y1, float r,float g,float b){
    ensure_overlay();
    float verts[8] = {
        px_to_ndc_x(x0), px_to_ndc_y(y0),
        px_to_ndc_x(x1), px_to_ndc_y(y0),
        px_to_ndc_x(x1), px_to_ndc_y(y1),
        px_to_ndc_x(x0), px_to_ndc_y(y1)
    };
    GLuint vao,vbo; glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo); glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,(void*)0);
    glUseProgram(ovProg);
    glUniform3f(ov_uColor,r,g,b);
    glDisable(GL_DEPTH_TEST);
    glLineWidth(2.0f);
    glDrawArrays(GL_LINE_LOOP,0,4);
    glLineWidth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);
    glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao);
}

// ---------------- HUD HELPERS (глобально, вне main) ----------------
void draw_filled_rect_pixels(float x0, float y0, float x1, float y1, float r, float g, float b, float a){
    float ndc[8] = {
        px_to_ndc_x(x0), px_to_ndc_y(y1),
        px_to_ndc_x(x1), px_to_ndc_y(y1),
        px_to_ndc_x(x1), px_to_ndc_y(y0),
        px_to_ndc_x(x0), px_to_ndc_y(y0)
    };
    float verts[12] = {
        ndc[0], ndc[1],
        ndc[2], ndc[3],
        ndc[4], ndc[5],
        ndc[0], ndc[1],
        ndc[4], ndc[5],
        ndc[6], ndc[7]
    };
    GLuint vao,vbo; glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo); glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,(void*)0);
    glUseProgram(ovProg);
    glUniform3f(ov_uColor, r, g, b);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLES,0,6);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBindBuffer(GL_ARRAY_BUFFER,0); glBindVertexArray(0);
    glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao);
}

void draw_digit7_px(float x, float y, float h, int val, float r, float g, float b){
    static const int segmap[10] = {
        0b00111111,0b00000110,0b01011011,0b01001111,0b01100110,
        0b01101101,0b01111101,0b00000111,0b01111111,0b01101111
    };
    if(val<0) val=0; if(val>9) val=9;
    int m = segmap[val];
    float w = h * 0.62f;
    float thickness = std::max(2.0f, h * 0.12f);
    auto seg = [&](float rx0, float ry0, float rw, float rh){
        draw_filled_rect_pixels(rx0, ry0, rx0+rw, ry0+rh, r,g,b,1.0f);
    };
    if(m & 0x01) seg(x + thickness, y, w - 2*thickness, thickness); // a
    if(m & 0x02) seg(x + w - thickness, y + thickness, thickness, (h - 3*thickness)/2.0f); // b
    if(m & 0x04) seg(x + w - thickness, y + thickness + (h - 3*thickness)/2.0f, thickness, (h - 3*thickness)/2.0f); // c
    if(m & 0x08) seg(x + thickness, y + h - thickness, w - 2*thickness, thickness); // d
    if(m & 0x10) seg(x, y + thickness + (h - 3*thickness)/2.0f, thickness, (h - 3*thickness)/2.0f); // e
    if(m & 0x20) seg(x, y + thickness, thickness, (h - 3*thickness)/2.0f); // f
    if(m & 0x40) seg(x + thickness, y + (h - thickness)*0.5f, w - 2*thickness, thickness); // g
}

float draw_number_px(float x, float y, float h, int number, float r, float g, float b){
    if(number < 0) number = 0;
    std::string s = std::to_string(number);
    float w = h * 0.62f;
    float spacing = std::max(2.0f, h * 0.08f);
    float cur = x;
    for(char ch : s){
        int d = ch - '0';
        draw_digit7_px(cur, y, h, d, r, g, b);
        cur += w + spacing;
    }
    return cur - x;
}
// ---------------- end HUD HELPERS ----------------

// HUD / FPS state (declare before main)
static int hud_fps_frames = 0;
static float hud_fps_acc = 0.0f;
static float hud_fps_value = 0.0f;
static const float HUD_FPS_UPDATE_INTERVAL = 0.25f; // update every 0.25s
static const int SCORE_BAR_MAX = 50; // при каком счёте бар считается полным

GLuint compile_shader(GLenum t, const char* src){
    GLuint s = glCreateShader(t);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ char buf[8192]; glGetShaderInfoLog(s,8192,nullptr,buf); std::cerr<<"Shader compile err: "<<buf<<"\n"; }
    return s;
}
GLuint link_program(GLuint vs, GLuint fs){
    GLuint p = glCreateProgram();
    glAttachShader(p,vs); glAttachShader(p,fs);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ char buf[8192]; glGetProgramInfoLog(p,8192,nullptr,buf); std::cerr<<"Link err: "<<buf<<"\n"; }
    return p;
}

int main(){
    if(!glfwInit()){ std::cerr<<"glfwInit failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = nullptr;
    if(primary) mode = glfwGetVideoMode(primary);

    int create_w = 1280;
    int create_h = 720;
    if(mode){
        create_w = mode->width;
        create_h = mode->height;
    }

    // fullscreen window on primary monitor
    GLFWwindow* wnd = glfwCreateWindow(create_w, create_h, "Flappy Whale 3D", primary, nullptr);
    if(!wnd){ std::cerr<<"window create failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(wnd);
    glfwSetKeyCallback(wnd, key_cb);
    auto framebuffer_size_callback = [](GLFWwindow* window, int width, int height){
        if(width <= 0 || height <= 0) return;
        WIN_W = width;
        WIN_H = height;
        glViewport(0, 0, WIN_W, WIN_H);
    };
    glfwSetFramebufferSizeCallback(wnd, framebuffer_size_callback);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){ std::cerr<<"glad failed\n"; return -1; }
    int fbw = create_w, fbh = create_h;
    glfwGetFramebufferSize(wnd, &fbw, &fbh);
    WIN_W = fbw; WIN_H = fbh;
    glViewport(0,0,WIN_W,WIN_H);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_deform_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_phong_src);
    GLuint prog_phong = link_program(vs,fs);
    glDeleteShader(vs); glDeleteShader(fs);

    GLint p_uModel = glGetUniformLocation(prog_phong, "uModel");
    GLint p_uTime = glGetUniformLocation(prog_phong, "uTime");
    GLint p_uLightDir = glGetUniformLocation(prog_phong, "uLightDir");
    GLint p_uColor = glGetUniformLocation(prog_phong, "uColor");
    GLint p_uCamPos = glGetUniformLocation(prog_phong, "uCamPos");
    GLint p_uSpecPow = glGetUniformLocation(prog_phong, "uSpecularPower");
    GLint p_uSpecInt = glGetUniformLocation(prog_phong, "uSpecularIntensity");

    ensure_overlay();

    Mesh cube = make_cube_mesh();

    Model whaleModel, stalactiteModel;
    bool whale_loaded=false, stal_loaded=false;
    auto find_asset_file = [](const std::string &relpath)->std::string{
        namespace fs = std::filesystem;
        std::vector<std::string> candidates = { relpath, std::string("./")+relpath, std::string("../")+relpath, std::string("assets/")+relpath, std::string("../assets/")+relpath };
        for(auto &c: candidates) if(fs::exists(c)) return c;
        fs::path p(relpath);
        if(p.has_filename()){
            std::string fn = p.filename().string();
            if(fs::exists(fn)) return fn;
        }
        return std::string();
    };

    std::string wp = find_asset_file("cute_whale.glb");
    if(wp.empty()) wp = find_asset_file("assets/cute_whale.glb");
    if(!wp.empty()){
        if(load_gltf_model(wp, whaleModel)){ whale_loaded = true; std::cerr<<"Loaded whale GLB: "<<wp<<"\n"; }
        else std::cerr<<"Failed to parse whale: "<<wp<<"\n";
    } else std::cerr<<"Cannot find cute_whale.glb — using cube fallback\n";

    std::string sp = find_asset_file("low-poly_stalactite.glb");
    if(sp.empty()) sp = find_asset_file("assets/low-poly_stalactite.glb");
    if(!sp.empty()){
        if(load_gltf_model(sp, stalactiteModel)){ stal_loaded = true; std::cerr<<"Loaded stalactite GLB: "<<sp<<"\n"; }
        else std::cerr<<"Failed to parse stalactite: "<<sp<<"\n";
    } else std::cerr<<"Cannot find low-poly_stalactite.glb — using cube fallback for pipes\n";

    glGenBuffers(1, &g_globalUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, g_globalUBO);
    glBufferData(GL_UNIFORM_BUFFER, 128, nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, g_globalUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint ubIndex = glGetUniformBlockIndex(prog_phong, "Global");
    if(ubIndex != GL_INVALID_INDEX) glUniformBlockBinding(prog_phong, ubIndex, 0);

    GLuint w_vs = compile_shader(GL_VERTEX_SHADER, water_vs);
    GLuint w_fs = compile_shader(GL_FRAGMENT_SHADER, water_fs);
    GLuint prog_water = link_program(w_vs,w_fs);
    glDeleteShader(w_vs); glDeleteShader(w_fs);
    Mesh water = make_water_grid(160, 120, 2400.0f, 1800.0f);

    pipes.clear();
    for(int i=0;i<INITIAL_PIPES;++i){
        float px = WIN_W*0.5f;
        int gapTop = rand_int(80, WIN_H - 80 - (int)GAP_HEIGHT);
        float gapCenterY = gapTop + GAP_HEIGHT*0.5f;
        float z = SPAWN_Z + i * PIPE_SPACING_Z;
        pipes.emplace_back(px, gapCenterY, z, g_nextPipeId++);
    }

    auto last = std::chrono::high_resolution_clock::now();
    auto startTime = last;
    float accumulator = 0.f;
    const float FIXED_DT = 1.0f/60.0f;

    Mat4 proj = mat4_perspective(60.0f * PI/180.0f, (float)WIN_W/(float)WIN_H, 0.1f, 5000.0f);

    // camera follow config
    const float CAM_EYE_Z = -350.0f;          // camera distance from scene (negative - away on -Z)
    const float CAM_OFFSET_X = 240.0f;  // how much camera is ahead of whale in X
    const float CAM_OFFSET_Y = 40.0f;   // vertical offset of camera relative to whale
    const float CAM_LOOK_OFFSET_X = 30.0f; // look-at offset X relative to whale
    const float CAM_LOOK_Z = 300.0f;    // look-at Z (depth in front)
    const float CAM_FOLLOW_LERP = 0.12f; // 0..1 smoothing of camera following (1.0 instant)

    // camera smoothing state
    float cam_eye_x = WIN_W*0.5f + CAM_OFFSET_X;
    float cam_eye_y = WIN_H*0.5f + CAM_OFFSET_Y;
    float cam_eye_z = CAM_EYE_Z;

    Mat4 view = mat4_lookat(cam_eye_x, cam_eye_y, cam_eye_z,
                            WIN_W*0.5f + CAM_LOOK_OFFSET_X, WIN_H*0.5f, CAM_LOOK_Z,
                            0.0f,1.0f,0.0f);
    Mat4 PV = mat4_mul(proj, view);

    whale.x = WIN_W*0.5f; whale.z = 0.0f; whale.y = WIN_H*0.5f;
    glClearColor(0.4f,0.75f,0.95f,1.0f);

    while(!glfwWindowShouldClose(wnd)){
        int cur_w, cur_h;
        glfwGetFramebufferSize(wnd, &cur_w, &cur_h);
        if(cur_w != WIN_W || cur_h != WIN_H){
            WIN_W = cur_w; WIN_H = cur_h;
            glViewport(0,0,WIN_W,WIN_H);
            proj = mat4_perspective(60.0f * PI/180.0f, (float)WIN_W/(float)WIN_H, 0.1f, 5000.0f);
        }

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> dt = now - last;
        last = now;
        float fd = dt.count();
        if(fd>0.1f) fd = 0.1f;
        accumulator += fd;

        glfwPollEvents();

        if(wantJump){
            if(!whale.alive){
                whale.reset();
                pipes.clear();
                for(int i=0;i<INITIAL_PIPES;++i){
                    int gapTop = rand_int(80, WIN_H - 80 - (int)GAP_HEIGHT);
                    float gapCenterY = gapTop + GAP_HEIGHT*0.5f;
                    float z = SPAWN_Z + i * PIPE_SPACING_Z;
                    pipes.emplace_back(WIN_W*0.5f, gapCenterY, z, g_nextPipeId++);
                }
                score = 0;
            }
            whale.vy = JUMP_V;
            wantJump = false;
        }

        while(accumulator >= FIXED_DT){
            if(whale.alive){
                whale.vy += GRAVITY * FIXED_DT;
                whale.y += whale.vy * FIXED_DT;
                if(whale.y - WHALE_RADIUS < 0.0f){ whale.y = WHALE_RADIUS; whale.vy = 0.0f; whale.alive=false; }
                if(whale.y + WHALE_RADIUS > WIN_H){ whale.y = WIN_H - WHALE_RADIUS; whale.vy = 0.0f; whale.alive=false; }
                for(auto &p: pipes) { p.z -= PIPE_SPEED * FIXED_DT; }
                if(!pipes.empty() && pipes.front().z < DESPAWN_Z){
                    pipes.erase(pipes.begin());
                    int gapTop = rand_int(80, WIN_H - 80 - (int)GAP_HEIGHT);
                    float gapCenterY = gapTop + GAP_HEIGHT*0.5f;
                    float candidateNewZ = pipes.back().z + PIPE_SPACING_Z;
                    float minSafeZ = whale.z + SPAWN_Z * 0.5f;
                    float newZ = std::max(candidateNewZ, minSafeZ);
                    pipes.emplace_back(WIN_W*0.5f, gapCenterY, newZ, g_nextPipeId++);
                    pipes.back().passed = false;
                }
                for(auto &p: pipes){
                    if(!p.passed && p.z < whale.z){
                        p.passed = true; score++; if(score>highScore) highScore = score;
                    }
                }
            }
            accumulator -= FIXED_DT;
        }

        // --- Camera follow: compute desired eye/look positions based on whale and smooth them ---
        float desired_eye_x = whale.x + CAM_OFFSET_X;
        float desired_eye_y = whale.y + CAM_OFFSET_Y;
        float desired_eye_z = CAM_EYE_Z; // keep fixed depth

        // smooth lerp
        cam_eye_x += (desired_eye_x - cam_eye_x) * CAM_FOLLOW_LERP;
        cam_eye_y += (desired_eye_y - cam_eye_y) * CAM_FOLLOW_LERP;
        cam_eye_z = desired_eye_z; // depth instant

        float look_x = whale.x + CAM_LOOK_OFFSET_X;
        float look_y = whale.y;
        float look_z = CAM_LOOK_Z;

        view = mat4_lookat(cam_eye_x, cam_eye_y, cam_eye_z,
                           look_x, look_y, look_z,
                           0.0f,1.0f,0.0f);

        PV = mat4_mul(proj, view);
        float timeSec = (float)std::chrono::duration<double>(now - startTime).count();

        struct GlobalUBO { float pv[16]; float light[4]; float cam[4]; };
        GlobalUBO gdata;
        memcpy(gdata.pv, PV.m, sizeof(float)*16);
        gdata.light[0] = 0.3f; gdata.light[1] = -1.0f; gdata.light[2] = 0.6f; gdata.light[3] = 0.0f;
        // camera position for shaders: must match eye position used in view
        gdata.cam[0] = cam_eye_x; gdata.cam[1] = cam_eye_y; gdata.cam[2] = cam_eye_z; gdata.cam[3] = 0.0f;
        glBindBuffer(GL_UNIFORM_BUFFER, g_globalUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(gdata), &gdata);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // water
        glUseProgram(prog_water);
        GLint w_uPV = glGetUniformLocation(prog_water,"uPV");
        GLint w_uModel = glGetUniformLocation(prog_water,"uModel");
        GLint w_uTime = glGetUniformLocation(prog_water,"uTime");
        GLint w_uCam = glGetUniformLocation(prog_water,"uCameraPos");
        GLint w_uLight = glGetUniformLocation(prog_water,"uLightDir");
        glUniformMatrix4fv(w_uPV,1,GL_FALSE,PV.m);
        Mat4 modelWater = mat4_mul(mat4_translate(WIN_W*0.5f, 0.0f, SPAWN_Z*0.5f), mat4_scale(1.0f,1.0f,1.0f));
        glUniformMatrix4fv(w_uModel,1,GL_FALSE, modelWater.m);
        glUniform1f(w_uTime, timeSec);
        glUniform3f(w_uCam, cam_eye_x, cam_eye_y, cam_eye_z);
        glUniform3f(w_uLight, 0.3f, -1.0f, 0.6f);
        glBindVertexArray(water.vao);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawElements(GL_TRIANGLES, water.idxCount, GL_UNSIGNED_INT, 0);
        glDisable(GL_BLEND);
        glBindVertexArray(0);

        // models: phong shader
        glUseProgram(prog_phong);
        glUniform1f(p_uTime, timeSec);
        glUniform3f(p_uLightDir, 0.3f, -1.0f, 0.6f);
        glUniform3f(p_uCamPos, cam_eye_x, cam_eye_y, cam_eye_z);
        glUniform1f(p_uSpecPow, 32.0f);
        glUniform1f(p_uSpecInt, 1.2f);

        GLboolean cullWasEnabled = glIsEnabled(GL_CULL_FACE);
        if(cullWasEnabled) glDisable(GL_CULL_FACE);

        float mhx=1.0f,mhy=1.0f,mhz=1.0f;
        if(stal_loaded) model_half_extents(stalactiteModel, mhx,mhy,mhz);
        else { mhx = mhy = mhz = 1.0f; }

        // draw stalactites so that visuals MATCH hitboxes (HITBOX_SCALE) and orientation is inverted (top up, bottom down)
        for(size_t i=0;i<pipes.size();++i){
            auto &p = pipes[i];

            float gapTopEdge = p.y + GAP_HEIGHT * 0.5f;
            float gapBottomEdge = p.y - GAP_HEIGHT * 0.5f;

            float top_height = WIN_H - gapTopEdge; if(top_height < 1.0f) top_height = 1.0f;
            float top_cy = gapTopEdge + top_height * 0.5f;

            float bottom_height = gapBottomEdge - 0.0f; if(bottom_height < 1.0f) bottom_height = 1.0f;
            float bottom_cy = bottom_height * 0.5f;

            float desired_hx = PIPE_HALF_WIDTH * HITBOX_SCALE;
            float desired_hy_top = (top_height * 0.5f) * HITBOX_SCALE;
            float desired_hy_bot = (bottom_height * 0.5f) * HITBOX_SCALE;
            float desired_hz = (PIPE_DEPTH * 0.5f) * HITBOX_SCALE;

            float scaleX_top = (mhx > 1e-6f) ? (desired_hx / mhx) * GLOBAL_STAL_SCALE : GLOBAL_STAL_SCALE * HITBOX_SCALE;
            float scaleY_top = (mhy > 1e-6f) ? (desired_hy_top / mhy) * GLOBAL_STAL_SCALE : GLOBAL_STAL_SCALE * HITBOX_SCALE;
            float scaleZ_top = (mhz > 1e-6f) ? (desired_hz / mhz) * GLOBAL_STAL_SCALE : GLOBAL_STAL_SCALE * HITBOX_SCALE;

            float scaleX_bot = scaleX_top;
            float scaleY_bot = (mhy > 1e-6f) ? (desired_hy_bot / mhy) * GLOBAL_STAL_SCALE : GLOBAL_STAL_SCALE * HITBOX_SCALE;
            float scaleZ_bot = scaleZ_top;

            if(p.z > DESPAWN_Z && p.z < SPAWN_Z + 800.0f){
                Mat4 rotTop = mat4_identity(); // top points UP now (no rotation)
                Mat4 modelTop = mat4_mul(mat4_translate(p.x, top_cy, p.z),
                                         mat4_mul(rotTop, mat4_scale(scaleX_top, scaleY_top, scaleZ_top)));
                glUniformMatrix4fv(p_uModel,1,GL_FALSE, modelTop.m);
                glUniform3f(p_uColor, 0.5f, 0.5f, 0.55f);
                if(stal_loaded){ glBindVertexArray(stalactiteModel.vao); glDrawElements(GL_TRIANGLES, stalactiteModel.idxCount, GL_UNSIGNED_INT, 0); }
                else { glBindVertexArray(cube.vao); glDrawElements(GL_TRIANGLES, cube.idxCount, GL_UNSIGNED_INT, 0); }
                glBindVertexArray(0);
            }

            if(p.z > DESPAWN_Z && p.z < SPAWN_Z + 800.0f){
                Mat4 rotBot = mat4_rotate_x(PI); // bottom points DOWN now
                Mat4 modelBot = mat4_mul(mat4_translate(p.x, bottom_cy, p.z),
                                         mat4_mul(rotBot, mat4_scale(scaleX_bot, scaleY_bot, scaleZ_bot)));
                glUniformMatrix4fv(p_uModel,1,GL_FALSE, modelBot.m);
                glUniform3f(p_uColor, 0.5f, 0.5f, 0.55f);
                if(stal_loaded){ glBindVertexArray(stalactiteModel.vao); glDrawElements(GL_TRIANGLES, stalactiteModel.idxCount, GL_UNSIGNED_INT, 0); }
                else { glBindVertexArray(cube.vao); glDrawElements(GL_TRIANGLES, cube.idxCount, GL_UNSIGNED_INT, 0); }
                glBindVertexArray(0);
            }
        }

        // whale draw (unchanged)
        float collision_radius = WHALE_RADIUS * 0.9f;
        float sphere_z = whale.z;
        if(whale_loaded){
            float mhx_w,mhy_w,mhz_w; model_half_extents(whaleModel,mhx_w,mhy_w,mhz_w);
            if(mhx_w < 1e-6f) mhx_w = 1.0f; if(mhy_w < 1e-6f) mhy_w = 1.0f; if(mhz_w < 1e-6f) mhz_w = 1.0f;
            float desired_half = WHALE_RADIUS * 0.8f;
            float sx = desired_half / mhx_w;
            float sy = desired_half / mhy_w;
            float sz = desired_half / mhz_w;
            Mat4 model = mat4_mul(mat4_translate(whale.x, whale.y, sphere_z), mat4_scale(sx,sy,sz));
            glUniformMatrix4fv(p_uModel,1,GL_FALSE,model.m);
            if(whale.alive)
                glUniform3f(p_uColor, 0.05f, 0.12f, 0.25f);   // dark black-blue whale
            else glUniform3f(p_uColor, 0.6f,0.1f,0.1f);
            glBindVertexArray(whaleModel.vao);
            glDrawElements(GL_TRIANGLES, whaleModel.idxCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
            float scaled_hx = mhx_w * sx;
            float scaled_hy = mhy_w * sy;
            collision_radius = std::max(scaled_hx, scaled_hy) * 0.9f;
        } else {
            Mat4 model = mat4_mul(mat4_translate(whale.x, whale.y, sphere_z), mat4_scale(WHALE_RADIUS*0.8f, WHALE_RADIUS*0.8f, WHALE_RADIUS*0.8f));
            glUniformMatrix4fv(p_uModel,1,GL_FALSE,model.m);
            if(whale.alive) glUniform3f(p_uColor, 0.95f, 0.8f, 0.25f);
            else glUniform3f(p_uColor, 0.6f, 0.1f, 0.1f);
            glBindVertexArray(cube.vao);
            glDrawElements(GL_TRIANGLES, cube.idxCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        if(cullWasEnabled) glEnable(GL_CULL_FACE);

        // collision: build AABBs from exact centers & sizes used for drawing (so they match visuals)
        if(whale.alive){
            for(auto &p : pipes){
                float gapTopEdge = p.y + GAP_HEIGHT * 0.5f;
                float gapBottomEdge = p.y - GAP_HEIGHT * 0.5f;

                float top_height = WIN_H - gapTopEdge; if(top_height < 1.0f) top_height = 1.0f;
                float top_cy = gapTopEdge + top_height * 0.5f;

                float bottom_height = gapBottomEdge - 0.0f; if(bottom_height < 1.0f) bottom_height = 1.0f;
                float bottom_cy = bottom_height * 0.5f;

                float desired_hx_vis = PIPE_HALF_WIDTH * HITBOX_SCALE;
                float desired_hy_top_vis = (top_height * 0.5f) * HITBOX_SCALE;
                float desired_hy_bot_vis = (bottom_height * 0.5f) * HITBOX_SCALE;
                float desired_hz_vis = (PIPE_DEPTH * 0.5f) * HITBOX_SCALE;

                float top_hx = desired_hx_vis;
                float top_hy = desired_hy_top_vis;
                float top_hz = desired_hz_vis;

                float bot_hx = desired_hx_vis;
                float bot_hy = desired_hy_bot_vis;
                float bot_hz = desired_hz_vis;

                AABB topA = { p.x, top_cy, p.z, top_hx, top_hy, top_hz };
                AABB botA = { p.x, bottom_cy, p.z, bot_hx, bot_hy, bot_hz };

                float sphere_z_local = sphere_z;
                if(sphere_aabb_collide(whale.x, whale.y, sphere_z_local, collision_radius, topA) ||
                   sphere_aabb_collide(whale.x, whale.y, sphere_z_local, collision_radius, botA))
                {
                    whale.alive = false;
                    break;
                }
            }
        }

        // overlay hitboxes (draw the exact reduced hitboxes so overlay == collision)
        if(showHitboxes){
            for(auto &p : pipes){
                float gapTopEdge = p.y + GAP_HEIGHT * 0.5f;
                float gapBottomEdge = p.y - GAP_HEIGHT * 0.5f;

                float top_height = WIN_H - gapTopEdge; if(top_height < 1.0f) top_height = 1.0f;
                float top_cy = gapTopEdge + top_height * 0.5f;

                float bottom_height = gapBottomEdge - 0.0f; if(bottom_height < 1.0f) bottom_height = 1.0f;
                float bottom_cy = bottom_height * 0.5f;

                float top_hx = PIPE_HALF_WIDTH * HITBOX_SCALE;
                float top_hy = (top_height * 0.5f) * HITBOX_SCALE;
                float top_hz = (PIPE_DEPTH * 0.5f) * HITBOX_SCALE;

                float bot_hx = PIPE_HALF_WIDTH * HITBOX_SCALE;
                float bot_hy = (bottom_height * 0.5f) * HITBOX_SCALE;
                float bot_hz = (PIPE_DEPTH * 0.5f) * HITBOX_SCALE;

                AABB topA = { p.x, top_cy, p.z, top_hx, top_hy, top_hz };
                AABB botA = { p.x, bottom_cy, p.z, bot_hx, bot_hy, bot_hz };

                ScreenRect tr = aabb_to_screen_rect(PV, topA);
                ScreenRect br = aabb_to_screen_rect(PV, botA);
                draw_screen_rect_lines(tr.x0,tr.y0,tr.x1,tr.y1, 1.0f,0.1f,0.1f);
                draw_screen_rect_lines(br.x0,br.y0,br.x1,br.y1, 1.0f,0.1f,0.1f);
            }
            float sx,sy,sz;
            if(world_to_screen(PV, whale.x, whale.y, whale.z, sx,sy,sz)){
                float sx2,sy2,sz2;
                world_to_screen(PV, whale.x + WHALE_RADIUS*0.8f, whale.y, whale.z, sx2,sy2,sz2);
                float rr = sqrtf((sx - sx2)*(sx - sx2) + (sy - sy2)*(sy - sy2));
                const int SEG = 32;
                std::vector<float> verts; verts.reserve(SEG*2);
                for(int i=0;i<SEG;++i){
                    float a = (float)i/(float)SEG * PI*2.0f;
                    float px = sx + cosf(a) * rr;
                    float py = sy + sinf(a) * rr;
                    verts.push_back(px_to_ndc_x(px));
                    verts.push_back(px_to_ndc_y(py));
                }
                GLuint vao,vbo; glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
                glBindVertexArray(vao);
                glBindBuffer(GL_ARRAY_BUFFER,vbo); glBufferData(GL_ARRAY_BUFFER, verts.size()*sizeof(float), verts.data(), GL_DYNAMIC_DRAW);
                glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,(void*)0);
                glUseProgram(ovProg); glUniform3f(ov_uColor, 1.0f, 0.95f, 0.15f);
                glDisable(GL_DEPTH_TEST); glLineWidth(2.0f);
                glDrawArrays(GL_LINE_LOOP, 0, SEG);
                glLineWidth(1.0f); glEnable(GL_DEPTH_TEST);
                glBindBuffer(GL_ARRAY_BUFFER,0); glBindVertexArray(0);
                glDeleteBuffers(1,&vbo); glDeleteVertexArrays(1,&vao);
            }
        }

        // ---------- HUD render (score bar + FPS) ----------
        {
            // update FPS
            hud_fps_frames += 1;
            hud_fps_acc += fd; // fd — у тебя в main уже есть
            if(hud_fps_acc >= HUD_FPS_UPDATE_INTERVAL){
                hud_fps_value = (float)hud_fps_frames / hud_fps_acc;
                hud_fps_frames = 0;
                hud_fps_acc = 0.0f;
            }

            // score bar (top-center)
            float bar_w = WIN_W * 0.42f;
            float bar_h = 20.0f;
            float bar_x0 = (WIN_W - bar_w) * 0.5f;
            float bar_y0 = 12.0f;
            float bar_x1 = bar_x0 + bar_w;
            float bar_y1 = bar_y0 + bar_h;

            // background
            draw_filled_rect_pixels(bar_x0, bar_y0, bar_x1, bar_y1, 0.05f, 0.05f, 0.05f, 0.9f);
            // fill proportionally
            float frac = (float)score / (float)std::max(1, SCORE_BAR_MAX);
            frac = std::max(0.0f, std::min(1.0f, frac));
            float fill_w = bar_w * frac;
            if(fill_w > 4.0f){
                draw_filled_rect_pixels(bar_x0 + 2.0f, bar_y0 + 2.0f, bar_x0 + 2.0f + (fill_w - 4.0f), bar_y1 - 2.0f, 0.12f, 0.82f, 0.20f, 0.95f);
            }
            // border
            draw_screen_rect_lines(bar_x0, bar_y0, bar_x1, bar_y1, 0.9f, 0.9f, 0.9f);

            // numeric score to the right of bar
            float score_x = bar_x1 + 16.0f;
            float score_y = bar_y0 - 2.0f;
            draw_number_px(score_x, score_y, 20.0f, score, 1.0f, 0.95f, 0.15f);

            // FPS (top-left)
            int fps_int = (int)std::round(hud_fps_value);
            draw_number_px(12.0f, 12.0f, 28.0f, fps_int, 1.0f, 0.85f, 0.12f);
        }
        // ---------- end HUD ----------

        std::string title = "Flappy Whale 3D  Score: " + std::to_string(score) + "  High: " + std::to_string(highScore);
        if(!whale.alive) title += "  (GAME OVER) press SPACE to restart";
        if(showHitboxes) title += "  [HITBOX ON]";
        glfwSetWindowTitle(wnd, title.c_str());

        glfwSwapBuffers(wnd);
    }

    destroy_model(whaleModel);
    destroy_model(stalactiteModel);
    glDeleteBuffers(1,&cube.vbo); glDeleteBuffers(1,&cube.ebo); glDeleteVertexArrays(1,&cube.vao);
    glDeleteBuffers(1,&water.vbo); glDeleteBuffers(1,&water.ebo); glDeleteVertexArrays(1,&water.vao);
    if(g_globalUBO) glDeleteBuffers(1,&g_globalUBO);
    glDeleteProgram(prog_phong);
    glDeleteProgram(prog_water);
    glfwTerminate();
    return 0;
}
