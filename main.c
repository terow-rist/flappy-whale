#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include "whale.h"
#include "pipes.h"

float gravity = -0.002f;
float lift = 0.05f;
float whaleY = 0.0f;
float velocity = 0.0f;
int gameOver = 0;

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Camera setup
    gluLookAt(0.0, 0.0, 5.0,   // eye (camera position)
              0.0, 0.0, 0.0,   // center (look-at point)
              0.0, 1.0, 0.0);  // up vector

    // Draw whale and pipes
    drawWhale(whaleY);
    drawPipes();

    glutSwapBuffers();
}

void update(int value) {
    if (!gameOver) {
        velocity += gravity;
        whaleY += velocity;

        updatePipes();

        if (checkCollision(whaleY)) {
            printf("Game Over!\n");
            gameOver = 1;
        }

        if (whaleY < -1.0f) {
            whaleY = -1.0f;
            velocity = 0;
            gameOver = 1;
        }
    }

    glutPostRedisplay();
    glutTimerFunc(16, update, 0);
}

void keyPress(unsigned char key, int x, int y) {
    if (key == 32 && !gameOver) { // Spacebar
        velocity = lift;
    } else if (key == 'r' && gameOver) {
        resetPipes();
        whaleY = 0;
        velocity = 0;
        gameOver = 0;
    } else if (key == 27) { // ESC
        exit(0);
    }
}

void reshape(int w, int h) {
    if (h == 0) h = 1;
    float aspect = (float)w / (float)h;

    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, aspect, 1.0, 20.0);

    glMatrixMode(GL_MODELVIEW);
}

void init() {
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.4f, 0.7f, 1.0f, 1.0f); // sky blue
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Flappy Whale 3D");

    init();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyPress);
    glutTimerFunc(16, update, 0);

    initPipes();
    glutMainLoop();
    return 0;
}
