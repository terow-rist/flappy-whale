#include <GL/glut.h>
#include "whale.h"

void drawWhale(float y) {
    glPushMatrix();
    glTranslatef(-1.2f, y, 0.0f);
    glColor3f(0.0, 0.3, 0.7);
    glutSolidSphere(0.2, 20, 20); // whale body

    // Tail
    glPushMatrix();
    glTranslatef(-0.25f, 0.0f, 0.0f);
    glScalef(0.3f, 0.1f, 1.0f);
    glutSolidCube(0.4f);
    glPopMatrix();

    glPopMatrix();
}
