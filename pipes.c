#include <GL/glut.h>
#include <stdlib.h>
#include "pipes.h"

#define MAX_PIPES 3
#define PIPE_GAP 0.6f
#define PIPE_SPEED 0.02f

typedef struct {
    float x;
    float height;
} Pipe;

Pipe pipes[MAX_PIPES];

void initPipes() {
    for (int i = 0; i < MAX_PIPES; i++) {
        pipes[i].x = 2.0f + i * 1.5f;
        pipes[i].height = ((float)rand() / RAND_MAX) * 0.8f - 0.4f;
    }
}

void resetPipes() {
    initPipes();
}

void updatePipes() {
    for (int i = 0; i < MAX_PIPES; i++) {
        pipes[i].x -= PIPE_SPEED;
        if (pipes[i].x < -2.0f) {
            pipes[i].x = 2.0f;
            pipes[i].height = ((float)rand() / RAND_MAX) * 0.8f - 0.4f;
        }
    }
}

void drawPipes() {
    glColor3f(0.0, 0.8, 0.1);
    for (int i = 0; i < MAX_PIPES; i++) {
        // Upper pipe
        glPushMatrix();
        glTranslatef(pipes[i].x, pipes[i].height + PIPE_GAP, 0.0f);
        glScalef(0.2f, 1.0f, 0.2f);
        glutSolidCube(0.8f);
        glPopMatrix();

        // Lower pipe
        glPushMatrix();
        glTranslatef(pipes[i].x, pipes[i].height - PIPE_GAP, 0.0f);
        glScalef(0.2f, 1.0f, 0.2f);
        glutSolidCube(0.8f);
        glPopMatrix();
    }
}

int checkCollision(float whaleY) {
    for (int i = 0; i < MAX_PIPES; i++) {
        float whaleX = -1.2f;
        if (pipes[i].x < whaleX + 0.2f && pipes[i].x > whaleX - 0.2f) {
            // Whale too high or too low within pipe gap
            if (whaleY > pipes[i].height + PIPE_GAP - 0.25f ||
                whaleY < pipes[i].height - PIPE_GAP + 0.25f)
                return 1;
        }
    }
    return 0;
}
