#include "obj_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>

// Very small OBJ loader: supports 'v', 'vn', 'f' (triangles)
int load_obj_mesh(const char* path, Mesh* out){
    FILE* f = fopen(path, "r");
    if(!f){ perror("open obj"); return 0;}
    float (*positions)[3] = NULL;
    float (*normals)[3] = NULL;
    int pos_cap=0, pos_cnt=0, nrm_cap=0, nrm_cnt=0;
    typedef struct { int p,n; } Index;
    Index* inds = NULL; int ind_cap=0, ind_cnt=0;
    char line[1024];
    while(fgets(line, sizeof(line), f)){
        if(line[0]=='v' && line[1]==' '){
            if(pos_cnt+1>pos_cap){ pos_cap = pos_cap? pos_cap*2: 512; positions = realloc(positions, pos_cap*3*sizeof(float)); }
            float x=0,y=0,z=0; sscanf(line+2, "%f %f %f", &x,&y,&z);
            positions[pos_cnt][0]=x; positions[pos_cnt][1]=y; positions[pos_cnt][2]=z; pos_cnt++;
        } else if(line[0]=='v' && line[1]=='n'){
            if(nrm_cnt+1>nrm_cap){ nrm_cap = nrm_cap? nrm_cap*2: 512; normals = realloc(normals, nrm_cap*3*sizeof(float)); }
            float x=0,y=0,z=0; sscanf(line+3, "%f %f %f", &x,&y,&z);
            normals[nrm_cnt][0]=x; normals[nrm_cnt][1]=y; normals[nrm_cnt][2]=z; nrm_cnt++;
        } else if(line[0]=='f' && line[1]==' '){
            int vi[3], vni[3];
            int c = sscanf(line+2, "%d//%d %d//%d %d//%d", &vi[0], &vni[0], &vi[1], &vni[1], &vi[2], &vni[2]);
            if(c<6){
                c = sscanf(line+2, "%d/%*d/%d %d/%*d/%d %d/%*d/%d", &vi[0], &vni[0], &vi[1], &vni[1], &vi[2], &vni[2]);
                if(c<6){
                    c = sscanf(line+2, "%d %d %d", &vi[0], &vi[1], &vi[2]);
                    if(c==3){ vni[0]=vni[1]=vni[2]=0; }
                }
            }
            for(int k=0;k<3;k++){
                if(ind_cnt+1>ind_cap){ ind_cap = ind_cap? ind_cap*2: 2048; inds = realloc(inds, ind_cap * sizeof(Index)); }
                inds[ind_cnt].p = vi[k]-1;
                inds[ind_cnt].n = vni[k]-1;
                ind_cnt++;
            }
        }
    }

    if(ind_cnt == 0 || pos_cnt == 0){
        free(positions); free(normals); free(inds);
        fclose(f);
        return 0;
    }

    // Build interleaved verts (pos + normal). If normals missing, set default normal.
    int vert_count = ind_cnt;
    float* verts = malloc(vert_count * 6 * sizeof(float));
    for(int i=0;i<ind_cnt;i++){
        int ip = inds[i].p;
        int in = inds[i].n;
        verts[i*6+0] = positions[ip][0];
        verts[i*6+1] = positions[ip][1];
        verts[i*6+2] = positions[ip][2];
        if(in>=0 && in < nrm_cnt){
            verts[i*6+3] = normals[in][0];
            verts[i*6+4] = normals[in][1];
            verts[i*6+5] = normals[in][2];
        } else {
            verts[i*6+3] = 0.0f;
            verts[i*6+4] = 0.0f;
            verts[i*6+5] = 1.0f;
        }
    }

    // Upload to VBO/VAO
    GLuint vbo, vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vert_count * 6 * sizeof(float), verts, GL_STATIC_DRAW);
    // positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(0));
    // normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));
    glBindVertexArray(0);

    out->vao = vao;
    out->vbo = vbo;
    out->count = vert_count;

    free(verts);
    free(positions);
    free(normals);
    free(inds);
    fclose(f);
    return 1;
}

void free_mesh(Mesh* m){
    if(!m) return;
    if(m->vbo) glDeleteBuffers(1, &m->vbo);
    if(m->vao) glDeleteVertexArrays(1, &m->vao);
    m->count = 0;
}
