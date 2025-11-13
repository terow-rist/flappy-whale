#include "shader.h"
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>

static char* slurp(const char* path) {
    FILE* f = fopen(path, "rb");
    if(!f) return NULL;
    fseek(f,0,SEEK_END);
    long s = ftell(f);
    fseek(f,0,SEEK_SET);
    char* buf = malloc(s+1);
    if(!buf){ fclose(f); return NULL;}
    fread(buf,1,s,f);
    buf[s]='\0';
    fclose(f);
    return buf;
}

static int compile_src(const char* src, GLenum type, GLuint* out) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader,1,&src,NULL);
    glCompileShader(shader);
    GLint ok=0; glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if(!ok){
        GLint len=0; glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        char* log = malloc(len+1);
        glGetShaderInfoLog(shader, len, NULL, log);
        fprintf(stderr, "Shader compile error: %s\n", log);
        free(log);
        glDeleteShader(shader);
        return 0;
    }
    *out = shader;
    return 1;
}

int compile_shader(const char* vert_path, const char* frag_path, unsigned int* out_program) {
    char* vsrc = slurp(vert_path);
    char* fsrc = slurp(frag_path);
    if(!vsrc || !fsrc){ fprintf(stderr, "Failed to read shader files\n"); free(vsrc); free(fsrc); return 0;}
    GLuint vsh, fsh;
    if(!compile_src(vsrc, GL_VERTEX_SHADER, &vsh)){ free(vsrc); free(fsrc); return 0;}
    if(!compile_src(fsrc, GL_FRAGMENT_SHADER, &fsh)){ glDeleteShader(vsh); free(vsrc); free(fsrc); return 0;}
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vsh);
    glAttachShader(prog, fsh);
    glLinkProgram(prog);
    GLint ok=0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if(!ok){
        GLint len=0; glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        char* log = malloc(len+1);
        glGetProgramInfoLog(prog, len, NULL, log);
        fprintf(stderr, "Program link error: %s\n", log);
        free(log);
        glDeleteShader(vsh); glDeleteShader(fsh); glDeleteProgram(prog);
        free(vsrc); free(fsrc);
        return 0;
    }
    glDeleteShader(vsh); glDeleteShader(fsh);
    free(vsrc); free(fsrc);
    *out_program = prog;
    return 1;
}
