#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H
#include <GL/glew.h>
typedef struct {
    GLuint vao, vbo;
    unsigned int count;
} Mesh;
int load_obj_mesh(const char* path, Mesh* out);
void free_mesh(Mesh* m);
#endif
