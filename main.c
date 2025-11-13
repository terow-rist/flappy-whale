#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include "shader.h"
#include "obj_loader.h"
#include "utils.h"

#define WIDTH 800
#define HEIGHT 600

static unsigned int program;
static Mesh whaleMesh = {0,0,0};
static Mesh coralMesh = {0,0,0};
static float whaleY = 0.0f;
static float velocity = 0.0f;
static float gravity = -0.002f;
static float lift = 0.05f;
static int gameOver = 0;
static float pipeX = 2.0f;

void drawMesh(Mesh* m, mat4 mvp, mat4 model, float r, float g, float b){
    glUseProgram(program);
    GLint loc = glGetUniformLocation(program, "uMVP");
    glUniformMatrix4fv(loc, 1, GL_FALSE, mvp.m);
    loc = glGetUniformLocation(program, "uModel");
    glUniformMatrix4fv(loc, 1, GL_FALSE, model.m);
    GLint light = glGetUniformLocation(program, "uLightPos");
    glUniform3f(light, 0.0f, 3.0f, 3.0f);
    GLint color = glGetUniformLocation(program, "uColor");
    glUniform3f(color, r, g, b);
    glBindVertexArray(m->vao);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)m->count);
    glBindVertexArray(0);
    glUseProgram(0);
}

void display(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mat4 proj = mat4_perspective(60.0f, (float)WIDTH/(float)HEIGHT, 0.1f, 100.0f);
    mat4 view = mat4_lookat(0.0f, 0.8f, 5.0f,  0.0f, 0.0f, 0.0f,   0.0f,1.0f,0.0f);

    // Whale transform (scale down if large)
    mat4 model_whale = mat4_mul(mat4_translate(-1.2f, whaleY, 0.0f), mat4_scale(0.6f,0.6f,0.6f));
    mat4 mvp = mat4_mul(proj, mat4_mul(view, model_whale));
    drawMesh(&whaleMesh, mvp, model_whale, 0.0f, 0.45f, 0.8f);

    // Coral (top & bottom pair) — draw two with gap
    mat4 model_coral_top = mat4_mul(mat4_translate(pipeX, 0.9f, 0.0f), mat4_scale(0.6f,0.6f,0.6f));
    mat4 mvp_top = mat4_mul(proj, mat4_mul(view, model_coral_top));
    drawMesh(&coralMesh, mvp_top, model_coral_top, 0.1f, 0.7f, 0.3f);

    mat4 model_coral_bottom = mat4_mul(mat4_translate(pipeX, -1.4f, 0.0f), mat4_scale(0.6f,0.6f,0.6f));
    mat4 mvp_bot = mat4_mul(proj, mat4_mul(view, model_coral_bottom));
    drawMesh(&coralMesh, mvp_bot, model_coral_bottom, 0.1f, 0.7f, 0.3f);

    glutSwapBuffers();
}

void update(int v){
    if(!gameOver){
        velocity += gravity;
        whaleY += velocity;
        pipeX -= 0.02f;
        if(pipeX < -3.5f) pipeX = 3.5f;

        // Basic AABB-ish collision with coral pair (very simple)
        float whaleX = -1.2f;
        float gap_center = -0.25f; // adjust as needed
        float gap_half = 0.5f;
        // If pipe is near whale X, test Y
        if(fabsf(pipeX - whaleX) < 0.5f){
            if(whaleY > gap_center + gap_half || whaleY < gap_center - gap_half){
                gameOver = 1;
                printf("Game Over\n");
            }
        }

        if(whaleY < -1.5f){ whaleY = -1.5f; velocity=0; gameOver=1; }
        if(whaleY > 1.6f){ whaleY = 1.6f; velocity=0; } // ceiling clamp
    }
    glutPostRedisplay();
    glutTimerFunc(16, update, 0);
}

void keyPress(unsigned char key, int x, int y){
    (void)x; (void)y;
    if(key == 32 && !gameOver){ velocity = lift; }
    else if(key == 'r' && gameOver){ whaleY=0; velocity=0; gameOver=0; pipeX=2.0f; }
    else if(key == 27) exit(0);
}

void reshape(int w, int h){
    glViewport(0,0,w,h);
}

int main(int argc, char** argv){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Flappy Whale - Modern");
    if(glewInit() != GLEW_OK){ fprintf(stderr,"GLEW init failed\n"); return 1; }
    glEnable(GL_DEPTH_TEST);

    if(!compile_shader("shaders/mesh.vert", "shaders/mesh.frag", &program)){ fprintf(stderr, "Shader compile failed\n"); return 1; }

    const char* whale_path = "assets/whale_simplified.obj";
    const char* coral_path = "assets/stalactite_simplified.obj";

    // mat4 model_whale = mat4_mul(mat4_translate(-1.2f, whaleY, 0.0f), mat4_scale(0.08f,0.08f,0.08f));


    if(!load_obj_mesh(whale_path, &whaleMesh)){
        fprintf(stderr, "Failed to load %s - falling back to triangle\n", whale_path);
        float fallback[18] = { 0,0,0,  0,0,1,  0.5,0,0,  0,0,1,  0,0.5,0,  0,0,1 };
        glGenVertexArrays(1, &whaleMesh.vao);
        glBindVertexArray(whaleMesh.vao);
        glGenBuffers(1, &whaleMesh.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, whaleMesh.vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(fallback), fallback, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));
        whaleMesh.count = 3;
    }

    if(!load_obj_mesh(coral_path, &coralMesh)){
        fprintf(stderr, "Failed to load %s - reusing whale fallback\n", coral_path);
        coralMesh = whaleMesh;
    }

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyPress);
    glutTimerFunc(16, update, 0);
    glClearColor(0.05f, 0.45f, 0.7f, 1.0f);
    glutMainLoop();
    return 0;
}
